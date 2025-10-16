import os
import torch
import argparse
import numpy as np
import wandb
import json
import urllib.request
import functools
import copy
from collections import OrderedDict

from tqdm import tqdm
from second_party.text_embedder.data.datasets import VideoMetadataDataset
from second_party.text_embedder.common.mmap import MemmapWriter

from transformers import GPT2LMHeadModel
from second_party.lavilla_narrator.lavila.models.tokenizer import MyGPT2Tokenizer
from second_party.lavilla_narrator.lavila.models.gpt2_gated import (
    GPT2LMHeadModel as GatedGPT2LMHeadModel,
)


def get_args_parser():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument(
        "--checkpoint-root",
        default="/ptmp/dduka/work/training_metadata/lavilla/checkpoints",
        type=str,
    )
    parser.add_argument(
        "--model-url",
        default="https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}",
        type=str,
    )
    parser.add_argument(
        "--model",
        default="vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth",
        type=str,
    )
    parser.add_argument("--gated-xattn", type=bool, default=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--video-metadata-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--flush-frequency", type=int, default=10)
    return parser


def augment_gpt2_config(config, cross_attn_freq=1, gated_xattn=True):
    new_config = copy.deepcopy(config)
    new_config.add_cross_attention = True
    new_config.add_cross_attention_freq = cross_attn_freq
    new_config.is_tanh_gating = gated_xattn
    return new_config


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def load_model_and_tokenizer(args, device):
    ckpt_path = os.path.join(args.checkpoint_root, args.model)

    os.makedirs(args.checkpoint_root, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print("Downloading model to {}".format(ckpt_path))
        urllib.request.urlretrieve(
            args.model_url.format(args.model),
            ckpt_path,
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    print(f"Checkpoint state_dict keys: {ckpt['state_dict'].keys()}")

    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v

    text_state_dict = {
        k.replace("text_decoder.", "", 1): v
        for k, v in state_dict.items()
        if k.startswith("text_decoder.")
    }

    gpt2 = GPT2LMHeadModel.from_pretrained(
        "gpt2-xl",
        use_cache=False,
    )

    new_config = augment_gpt2_config(
        gpt2.config, cross_attn_freq=3, gated_xattn=args.gated_xattn
    )
    
    model = GatedGPT2LMHeadModel(new_config)

    for n, p in gpt2.named_parameters():
        rsetattr(model, n + ".data", p.data)

    missing, unexpected = model.load_state_dict(text_state_dict, strict=True)
    print(
        f"Loaded text weights. Missing={missing[:3]}... Unexpected={unexpected[:3]}..."
    )

    tokenizer = MyGPT2Tokenizer(args.tokenizer_name, add_bos=True)

    with torch.no_grad():
        dim = model.encode_text(tokenizer(["foo"]).to(device)).shape[-1]
        print(f"Dimension of the text embeddings: {dim}")

    return model, tokenizer, dim


@torch.no_grad()
@torch.autocast("cuda")
def encode_text(model, text, normalize=True):
    # The text is already tokenized
    return model.encode_text(text, normalize=normalize)


# NOTE: Only the loading of the model is done!!! Take care of the rest!!!
def main(args):
    # wandb.init(
    #     project="Thesis",
    #     name=f"{args.model}_{args.video_metadata_path.split('/')[-2]}",
    #     config={**args.__dict__},
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer, dim = load_model_and_tokenizer(args, device)

    # Load video metadata dataset
    print(f"Loaded model and tokenizer")
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
        os.path.dirname(args.output_path), f"{args.model_name}_{args.pretrained}"
    )
    os.makedirs(output_dir, exist_ok=True)

    index_path = os.path.join(output_dir, "index.json")

    shape = (len(video_metadata_dataset), dim)

    writer = MemmapWriter(
        output_dir=output_dir,
        filename="embeddings.memmap",
        shape=shape,
        dtype=np.float32,
        mode="w+",
        flush_frequency=args.flush_frequency,
    )

    print(f"Estimated memory usage: {writer.estimated_megabytes():.2f} MB")

    index_array = {"captions": {}, "metadata": {**args.__dict__}}

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding text")):
        original_caption, caption, _ = (
            batch["original_caption"],
            batch["caption"],
            batch["frequency"],
        )

        caption = caption.to(device)

        text_features = encode_text(model, caption).detach().float().cpu().numpy()

        for i in range(len(original_caption)):
            global_index = batch_idx * args.batch_size + i
            index_array["captions"][original_caption[i]] = global_index

            writer.write_row(global_index, text_features[i])

        wandb.log(
            {
                "progress": (batch_idx + 1) / len(dataloader),
            }
        )

    writer.flush()

    with open(index_path, "w") as f:
        json.dump(index_array, f)


# Run the script using: python3 -m second_party.text_embedder.models.clip.clip
if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
