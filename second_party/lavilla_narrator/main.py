import os
import wandb
import argparse
import urllib.request
import avion.utils.distributed as dist_utils
from collections import OrderedDict

import torch
import json
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from lavila.data.datasets import VideoNarratorDataset
from lavila.data.video_transforms import Permute
from lavila.data.datasets import get_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from lavila.models.tokenizer import MyGPT2Tokenizer
from torch.utils.data._utils.collate import default_collate as _default_collate


def decode_one(generated_ids, tokenizer):
    # get the index of <EOS>
    if tokenizer.eos_token_id == tokenizer.bos_token_id:
        if tokenizer.eos_token_id in generated_ids[1:].tolist():
            eos_id = generated_ids[1:].tolist().index(tokenizer.eos_token_id) + 1
        else:
            eos_id = len(generated_ids.tolist()) - 1
    elif tokenizer.eos_token_id in generated_ids.tolist():
        eos_id = generated_ids.tolist().index(tokenizer.eos_token_id)
    else:
        eos_id = len(generated_ids.tolist()) - 1
    generated_text_str = tokenizer.tokenizer.decode(generated_ids[1:eos_id].tolist())
    return generated_text_str


def load_model_and_tokenizer(args):
    ckpt_path = os.path.join(args.checkpoint_root, args.model)

    os.makedirs(args.checkpoint_root, exist_ok=True)

    if not os.path.exists(ckpt_path):
        print("Downloading model to {}".format(ckpt_path))
        urllib.request.urlretrieve(
            args.model_url.format(args.model),
            ckpt_path,
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = OrderedDict()

    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v

    # Instantiate the model, and load the pre-trained weights
    model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
        text_use_cls_token=args.text_use_cls_token,
        project_embed_dim=args.project_embed_dim,
        gated_xattn=args.gated_xattn,
        timesformer_gated_xattn=args.timesformer_gated_xattn,
        freeze_lm_vclm=args.freeze_lm_vclm,
        freeze_visual_vclm=args.freeze_visual_vclm,
        num_frames=args.num_frames,
        drop_path_rate=args.drop_path_rate,
    )

    model.load_state_dict(state_dict, strict=True)

    # Assert that cuda is available
    assert torch.cuda.is_available(), "CUDA is not available"

    # Move model to the correct GPU based on rank
    if hasattr(args, "gpu") and args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    else:
        model.cuda()

    model.eval()

    tokenizer = MyGPT2Tokenizer(args.tokenizer_name, add_bos=True)

    return model, tokenizer


def load_val_transform(args):
    return transforms.Compose(
        [
            Permute([3, 0, 1, 2]),
            transforms.Resize(args.crop_size),
            transforms.CenterCrop(args.crop_size),
            transforms_video.NormalizeVideo(
                mean=[108.3272985, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316305],
            ),
        ]
    )


def generate_text(
    generated_text_ids, tokenizer, num_return_sequences, num_chunks_per_video
):
    generated_text_strs = []

    for i in range(generated_text_ids.shape[0]):
        generated_text_str = decode_one(generated_text_ids[i], tokenizer)
        generated_text_strs.append(generated_text_str)

    # Group the generated texts into chunks of num_return_sequences
    generated_text_strs = [
        generated_text_strs[i : i + num_return_sequences]
        for i in range(0, len(generated_text_strs), num_return_sequences)
    ]

    generated_text_strs = [
        generated_text_strs[i : i + num_chunks_per_video]
        for i in range(0, len(generated_text_strs), num_chunks_per_video)
    ]

    return generated_text_strs


def custom_collate_fn(batch):
    # batch: list of dicts from your dataset __getitem__
    # {
    #   "video_path": str,
    #   "frames": Tensor [S, T, C, H, W],
    #   "frames_ids": List[int] (or Tensor),
    #   "fps": float/int,
    #   "caption_path": str,
    #   "is_dummy_frame": bool,
    # }

    # Filter out samples with dummy frames
    batch = [b for b in batch if not b.get("is_dummy_frame", False)]

    # Handle case where all samples in the batch were filtered out
    if len(batch) == 0:
        return None

    out = {}

    out["frames"] = torch.stack(
        [b["frames"] for b in batch], dim=0
    )  # [B, S, C, T, H, W]

    B, S, C, T, H, W = out["frames"].shape
    out["frames"] = (
        out["frames"].reshape(B * S, C, T, H, W).contiguous()  # [B*S, C, T, H, W]
    )

    out["fps"] = torch.tensor([b["fps"] for b in batch])

    frames_ids_list = [b["frames_ids"] for b in batch]
    if isinstance(frames_ids_list[0], torch.Tensor):
        try:
            out["frames_ids"] = torch.stack(frames_ids_list, dim=0)
        except Exception:
            out["frames_ids"] = frames_ids_list
    else:
        try:
            out["frames_ids"] = _default_collate(frames_ids_list)
        except Exception:
            out["frames_ids"] = frames_ids_list

    out["video_path"] = [b["video_path"] for b in batch]
    out["caption_path"] = [b["caption_path"] for b in batch]

    return out


def main(args):
    # Initialize distributed mode
    dist_utils.init_distributed_mode(args)

    # Initialize wandb only on main process
    if dist_utils.is_main_process():
        wandb.init(
            project="Thesis", id=args.wandb_run_name, config=args, resume="allow"
        )

    model, tokenizer = load_model_and_tokenizer(args)

    val_transform = load_val_transform(args)

    dataset = VideoNarratorDataset(
        video_root=args.video_path_root,
        caption_suffix=f"lavila_captions_num_frames_{args.num_frames}/temperature_{args.temperature}",
        num_frames=args.num_frames,
        num_segments=args.num_segments,
        val_transform=val_transform,
        jitter=False,
    )

    print(f"len(dataset) = {len(dataset)}")

    # Create distributed sampler if in distributed mode
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=False
        )
    else:
        sampler = None

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        sampler=sampler,
    )

    print(f"len(dataloader) = {len(dataloader)}")

    for i, sample in enumerate(dataloader):
        # Skip None batches (when all samples had dummy frames)
        if sample is None:
            continue

        frames = sample["frames"]
        video_path = sample["video_path"]
        caption_path = sample["caption_path"]

        with torch.no_grad():
            # Move frames to the correct GPU
            if hasattr(args, "gpu") and args.gpu is not None:
                frames = frames.cuda(args.gpu, non_blocking=True)
            else:
                frames = frames.cuda(non_blocking=True)
            image_features = model.encode_image(frames)

            generated_text_ids, _ = model.generate(
                image_features,
                tokenizer,
                target=None,
                max_text_length=args.max_text_length,
                top_k=args.top_k,
                top_p=args.top_p,
                num_return_sequences=args.num_return_sequences,
                temperature=args.temperature,
                early_stopping=args.early_stopping,
            )

            # A list of lists with a size of batch_size
            generated_text_strs = generate_text(
                generated_text_ids,
                tokenizer,
                args.num_return_sequences,
                args.num_segments // args.num_frames,
            )

            frame_ids = sample["frames_ids"].tolist()
            fps = sample["fps"].tolist()

            # Store results for this batch. video_path has the length of a batch
            for j, (v_path, c_path) in enumerate(zip(video_path, caption_path)):
                # I want to have a caption file for each of the images. This means that it
                # would be better to loop over the video_path and captions path. Maybe
                # keeping track of the index is helpful too.
                results = {"video_path": v_path, "c_path": c_path, "metadata": []}

                frame_ids_for_video = frame_ids[j]
                generated_captions_for_video = generated_text_strs[j]
                fps_for_video = fps[j]

                for frames_id, captions in zip(
                    frame_ids_for_video, generated_captions_for_video
                ):
                    results["metadata"].append(
                        {
                            "captions": captions,
                            "frame_ids": frames_id,
                            "timestamps": [
                                frame_id / fps_for_video for frame_id in frames_id
                            ],  # Convert frame ids to timestamps,
                            "fps": fps_for_video,
                        }
                    )

                if not os.path.exists(f"{c_path}"):
                    os.makedirs(c_path, exist_ok=True)

                with open(f"{c_path}/captions.json", "w") as f:
                    json.dump(results, f)

            # Log progress only on main process
            if dist_utils.is_main_process():
                wandb.log({"progress": i / len(dataloader)})

        if i % 10 == 0 and i > 0:
            torch.cuda.empty_cache()


def get_args_parser():
    parser = argparse.ArgumentParser(description="LAVILA narrator", add_help=True)
    parser.add_argument("--wandb-project-name", default="Thesis", type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str, required=True)
    parser.add_argument("--wandb-log-video", action="store_true")

    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)

    # Distributed training parameters
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--distributed",
        default=False,
        action="store_true",
        help="enable distributed mode",
    )

    parser.add_argument(
        "--video-path-root",
        default="/ptmp/dduka/databases/ego4d/video_320px_15sec",
        type=str,
    )

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

    parser.add_argument("--tokenizer-name", default="gpt2-xl", type=str)

    parser.add_argument("--max-text-length", default=77, type=int)
    parser.add_argument("--num-return-sequences", default=10, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.95, type=float)
    parser.add_argument("--top-k", default=None, type=int)
    parser.add_argument("--project-embed-dim", default=256, type=int)
    parser.add_argument("--crop-size", default=336, type=int)
    parser.add_argument("--drop-path-rate", default=0.0, type=float)
    parser.add_argument("--text-use-cls-token", action="store_true")
    parser.add_argument("--freeze-lm-vclm", action="store_true")
    parser.add_argument("--freeze-visual-vclm", action="store_true")
    parser.add_argument("--gated-xattn", default=True, type=bool)
    parser.add_argument("--timesformer-gated-xattn", default=False, type=bool)

    parser.add_argument("--num-segments", default=4, type=int)
    parser.add_argument("--num-frames", default=4, type=int)

    parser.add_argument("--early-stopping", default=True, type=bool)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()

    try:
        main(args)
    except Exception as e:
        if dist_utils.is_main_process():
            print(f"Error during narration generation: {e}", flush=True)
        if dist_utils.is_dist_avail_and_initialized():
            torch.distributed.barrier()
        if dist_utils.is_main_process():
            wandb.finish(exit_code=1)
        raise
    finally:
        # Synchronize all processes before finishing
        if dist_utils.is_dist_avail_and_initialized():
            torch.distributed.barrier()
        if dist_utils.is_main_process():
            wandb.finish()
