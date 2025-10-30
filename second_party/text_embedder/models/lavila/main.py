import argparse
import json
import os
import numpy as np
import torch
import avion.models.model_clip as model_clip
import wandb
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
from collections import OrderedDict
from avion.data.tokenizer import tokenize
from avion.models.utils import inflate_positional_embeds
from second_party.text_embedder.data.datasets import VideoMetadataDataset


def get_args_parser():
    parser = argparse.ArgumentParser(description="LAVILA text embedder", add_help=False)
    parser.add_argument("--output-path", default="./", type=str, help="output dir")
    parser.add_argument(
        "--video-metadata-path", default="", type=str, help="Path to video metadata"
    )
    parser.add_argument(
        "--pretrain-model", default="", type=str, help="path of pretrained model"
    )
    parser.add_argument("--model-name", default="LAVILA", type=str)
    parser.add_argument(
        "--preprocess-function", default="preprocess_captions", type=str
    )

    # --- Dataloader Arguments ---
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="number of samples per-device/per-gpu",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers per process",
    )
    parser.add_argument("--flush-frequency", default=200, type=int)

    # --- Model Loading Arguments (Keep for compatibility with ckpt) ---
    parser.add_argument("--resume", default="", type=str, help="path to resume from")
    parser.add_argument("--model", default="CLIP_VITB16", type=str)
    parser.add_argument("--clip-length", default=16, type=int, help="clip length")
    parser.add_argument(
        "--grad-checkpointing", action="store_true", dest="use_grad_checkpointing"
    )
    parser.add_argument(
        "--no-grad-checkpointing", action="store_false", dest="use_grad_checkpointing"
    )
    parser.set_defaults(use_grad_checkpointing=False)
    parser.add_argument("--use-fast-conv1", action="store_true", dest="use_fast_conv1")
    parser.add_argument(
        "--disable-fast-conv1", action="store_false", dest="use_fast_conv1"
    )
    parser.set_defaults(use_fast_conv1=False)
    parser.add_argument("--use-flash-attn", action="store_true", dest="use_flash_attn")
    parser.add_argument(
        "--disable-flash-attn", action="store_false", dest="use_flash_attn"
    )
    parser.set_defaults(use_flash_attn=False)
    parser.add_argument("--patch-dropout", default=0.0, type=float)
    parser.add_argument("--drop-path-rate", default=0.0, type=float)
    parser.add_argument(
        "--freeze-temperature", action="store_true", dest="freeze_temperature"
    )
    parser.add_argument("--context-length", default=77, type=int)
    parser.add_argument("--vocab-size", default=49408, type=int)

    parser.add_argument("--project-embed-dim", default=256, type=int)
    parser.add_argument(
        "--pretrain-zoo",
        default="openai",
        type=str,
        choices=["openai", "open_clip", "lavila"],
    )
    parser.add_argument("--pretrain-path", default=None, type=str)

    return parser


def load_model(args, device):
    model = getattr(model_clip, args.model)(
        freeze_temperature=args.freeze_temperature,
        use_grad_checkpointing=args.use_grad_checkpointing,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        patch_dropout=args.patch_dropout,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
        use_fast_conv1=args.use_fast_conv1,
        use_flash_attn=args.use_flash_attn,
        use_quick_gelu=True,
        project_embed_dim=args.project_embed_dim,
        pretrain_zoo=args.pretrain_zoo,
        pretrain_path=args.pretrain_path,
    )
    model.to(device)

    if os.path.isfile(args.resume):
        print("=> loading resume checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint["args"].clip_length != args.clip_length:
            load_temporal_embedding = checkpoint["state_dict"][
                "module.visual.temporal_embedding"
            ]
            load_temporal_embedding = load_temporal_embedding.unsqueeze(0).permute(
                0, 2, 1
            )
            new_temporal_embed = (
                F.interpolate(
                    load_temporal_embedding, size=(args.clip_length,), mode="linear"
                )
                .permute(0, 2, 1)
                .squeeze(0)
            )
            checkpoint["state_dict"][
                "module.visual.temporal_embedding"
            ] = new_temporal_embed
        epoch = checkpoint["epoch"] if "epoch" in checkpoint else 0
        result = model.load_state_dict(checkpoint["state_dict"], strict=False)
        print(result)
        print("=> loaded resume checkpoint '{}' (epoch {})".format(args.resume, epoch))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    tokenizer = partial(tokenize, context_length=args.context_length)

    with torch.no_grad():
        encoded_input = tokenizer("foo").to(device)
        text_embeddings = model.encode_text(
            encoded_input, cast_dtype=torch.float32
        )
        dim = text_embeddings.shape[-1]
        print(f"Dimension of the text embeddings: {dim}")

    return model, tokenizer, dim


def main(args):
    wandb.init(
        project="Thesis",
        name=f"{args.model_name}_{args.video_metadata_path.split('/')[-2]}_{args.preprocess_function}",
        config={**args.__dict__},
        group=f"Text Embeddings",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, tokenizer, dim = load_model(args, device)

    torch.backends.cudnn.benchmark = True

    video_metadata_dataset = VideoMetadataDataset(args.video_metadata_path, tokenizer)
    print(f"Created video metadata dataset")

    # Create the dataloader
    print(f"Creating dataloader")
    dataloader = torch.utils.data.DataLoader(
        video_metadata_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    print(f"Created dataloader")

    output_dir = os.path.join(
        args.output_path,
        f"{args.model_name}",
        f"{args.preprocess_function}",
    )
    os.makedirs(output_dir, exist_ok=True)

    shape = (len(video_metadata_dataset), dim)
    print(f"Shape of the memmap: {shape}")
    print(f"Estimated memory usage: {shape[0] * shape[1] * 4 / 1024 / 1024:.2f} MB")

    memmap = np.memmap(
        os.path.join(output_dir, f"embeddings.memmap"),
        shape=shape,
        mode="w+",
        dtype=np.float32,
    )

    captions = {}

    # Ensure no gradients are computed
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding text")):
            idx, original_caption, caption, _ = (
                batch["idx"],
                batch["original_caption"],
                batch["caption"],
                batch["frequency"],
            )

            caption = caption.to(device)

            text_features = model.encode_text(caption, cast_dtype=torch.float32)
            text_features = F.normalize(text_features, dim=-1)
            text_features = text_features.float().cpu().numpy()

            for i in range(len(original_caption)):
                memmap[idx[i]] = text_features[i]

                if original_caption[i] in captions:
                    print(f"Warning: Caption {original_caption[i]} already exists")

                captions[original_caption[i]] = idx[i].item()

            if batch_idx % args.flush_frequency == 0 and batch_idx > 0:
                memmap.flush()

                wandb.log(
                    {
                        "progress": (batch_idx + 1) / len(dataloader),
                    }
                )

    memmap.flush()

    with open(os.path.join(output_dir, f"captions.json"), "w") as f:
        json.dump(captions, f)

    with open(os.path.join(output_dir, f"shape.json"), "w") as f:
        json.dump({"shape": list(shape)}, f)

    print(f"Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "LAVILA text embedding", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    main(args)
