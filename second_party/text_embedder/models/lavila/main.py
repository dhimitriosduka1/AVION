import argparse
import json
import os
import numpy as np
import torch
import avion.models.model_clip as model_clip
import wandb

from tqdm import tqdm
from functools import partial
from collections import OrderedDict
from avion.data.tokenizer import tokenize
from avion.models.utils import inflate_positional_embeds
from second_party.text_embedder.data.datasets import VideoMetadataDataset


def get_args_parser():
    parser = argparse.ArgumentParser(description="LAVILA text embedder", add_help=False)
    parser.add_argument("--output-dir", default="./", type=str, help="output dir")
    parser.add_argument("--video-chunk-length", default=15, type=int)
    parser.add_argument("--clip-length", default=16, type=int, help="clip length")
    parser.add_argument("--clip-stride", default=4, type=int, help="clip stride")
    parser.add_argument(
        "--norm-style", default="openai", type=str, choices=["openai", "timm"]
    )
    parser.add_argument(
        "--fused-decode-crop", action="store_true", dest="fused_decode_crop"
    )
    parser.add_argument(
        "--no-fused-decode-crop", action="store_false", dest="fused_decode_crop"
    )
    parser.set_defaults(fused_decode_crop=False)
    parser.add_argument("--decode-threads", default=1, type=int)
    parser.add_argument("--use-multi-epochs-loader", action="store_true")
    # model
    parser.add_argument("--model", default="CLIP_VITB16", type=str)
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
        "--pretrain-model", default="", type=str, help="path of pretrained model"
    )
    parser.add_argument("--resume", default="", type=str, help="path to resume from")
    # clip loss
    parser.add_argument("--local-loss", action="store_true")
    parser.add_argument(
        "--gather-with-grad", action="store_true", dest="gather_with_grad"
    )
    parser.add_argument(
        "--no-gather-with-grad", action="store_false", dest="gather_with_grad"
    )
    parser.set_defaults(gather_with_grad=True)
    # training
    parser.add_argument(
        "--use-zero", action="store_true", dest="use_zero", help="use ZeRO optimizer"
    )
    parser.add_argument(
        "--no-use-zero",
        action="store_false",
        dest="use_zero",
        help="use ZeRO optimizer",
    )
    parser.set_defaults(use_zero=False)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--warmup-epochs", default=1, type=int)
    parser.add_argument("--start-epoch", default=0, type=int)
    parser.add_argument(
        "--batch-size",
        default=16,
        type=int,
        help="number of samples per-device/per-gpu",
    )
    parser.add_argument("--optimizer", default="adamw", type=str)
    parser.add_argument("--lr", default=3e-5, type=float)
    parser.add_argument(
        "--lr-start", default=1e-6, type=float, help="initial warmup lr"
    )
    parser.add_argument("--lr-end", default=1e-5, type=float, help="minimum final lr")
    parser.add_argument(
        "--update-freq",
        default=1,
        type=int,
        help="optimizer update frequency (i.e. gradient accumulation steps)",
    )
    parser.add_argument("--wd", default=0.01, type=float)
    parser.add_argument("--betas", default=(0.9, 0.999), nargs=2, type=float)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--eval-freq", default=5, type=int)
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="disable mixed-precision training (requires more memory and compute)",
    )
    parser.add_argument("--grad-clip-norm", default=None, type=float)
    # system
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument(
        "-j",
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers per process",
    )
    parser.add_argument("--evaluate", action="store_true", help="eval only")
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")

    parser.add_argument("--model-name", default="LAVILA", type=str)
    parser.add_argument("--preprocess-function", default="preprocess_captions", type=str)
    parser.add_argument("--video-metadata-path", default="", type=str)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--num-workers", default=8, type=int)
    parser.add_argument("--flush-frequency", default=200, type=int)
    
    return parser


def load_model(args, device):
    if args.pretrain_model:
        ckpt_path = args.pretrain_model
    else:
        raise Exception(
            "no checkpoint found, add it by `--pretrain-model ${CHECKPOINT_PATH}`"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v

    old_args = ckpt["args"]
    print("=> creating model: {}".format(old_args.model))

    model = getattr(model_clip, old_args.model)(
        freeze_temperature=True,
        use_grad_checkpointing=args.use_grad_checkpointing,
        context_length=old_args.context_length,
        vocab_size=old_args.vocab_size,
        patch_dropout=args.patch_dropout,
        num_frames=args.clip_length,
        drop_path_rate=args.drop_path_rate,
        use_fast_conv1=args.use_fast_conv1,
        use_flash_attn=args.use_flash_attn,
        use_quick_gelu=True,
        project_embed_dim=old_args.project_embed_dim,
        pretrain_zoo=old_args.pretrain_zoo,
        pretrain_path=old_args.pretrain_path,
    )

    model.logit_scale.requires_grad = False
    print("=> inflating PE in models due to different frame numbers")

    state_dict = inflate_positional_embeds(
        model.state_dict(),
        state_dict,
        num_frames=args.clip_length,
        load_temporal_fix="bilinear",
    )

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    print(
        "=> loaded resume checkpoint '{}' (epoch {})".format(ckpt_path, ckpt["epoch"])
    )

    tokenizer = partial(tokenize, context_length=old_args.context_length)

    with torch.no_grad():
        encoded_input = tokenizer("foo")[0]
        encoded_input = encoded_input.to(device)
        text_embeddings = model.textual(encoded_input)
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

    n_wd, n_non_wd = [], []
    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if (
            p.ndim < 2
            or "bias" in n
            or "ln" in n
            or "bn" in n
            or "pos_embed" in n
            or "positional_embedding" in n
        ):
            n_non_wd.append(n)
            p_non_wd.append(p)
        else:
            n_wd.append(n)
            p_wd.append(p)

    torch.backends.cudnn.benchmark = True

    video_metadata_dataset = VideoMetadataDataset(args.video_metadata_path, None)
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
        os.path.dirname(args.output_path),
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
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding text")):
        idx, original_caption, caption, _ = (
            batch["idx"],
            batch["original_caption"],
            batch["caption"],
            batch["frequency"],
        )

        caption = caption.to(device)

        text_features = model.model.encode_text(caption)
        text_features = text_features.detach().float().cpu().numpy()

        for i in range(len(original_caption)):
            memmap[idx[i]] = text_features[i]

            if original_caption[i] in captions:
                print(f"Warning: Caption {original_caption[i]} already exists")

            captions[original_caption[i]] = idx[i]

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
        "LAVILA training and evaluation", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
