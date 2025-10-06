import os
import wandb
import argparse
import urllib.request
import avion.utils.distributed as dist_utils
from collections import OrderedDict

import torch
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


def generate_text(generated_text_ids, tokenizer, num_return_sequences):
    generated_text_strs = []

    for i in range(num_return_sequences):
        generated_text_str = decode_one(generated_text_ids[i], tokenizer)
        generated_text_strs.append(generated_text_str)

    return generated_text_strs


def load_frames(video_path, num_segments, num_frames, val_transform, jitter=False):
    original_frames, frame_ids, fps = get_frames(
        video_path=video_path, num_segments=num_segments, jitter=jitter
    )

    assert num_segments // num_frames == 15
    number_of_chunks = num_segments // num_frames

    original_frames_chunked = original_frames.chunk(number_of_chunks)
    frame_ids_chunked = np.array_split(frame_ids, number_of_chunks)

    frames_chunked = []
    for chunk in original_frames_chunked:
        frames_chunked.append(val_transform(chunk).unsqueeze(0))

    return frames_chunked, frame_ids_chunked, fps


def load_all_video_and_captions_paths(root_dir):
    video_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".mp4"):
                video_paths.append(os.path.abspath(os.path.join(dirpath, name)))

    # TODO: Maybe remove this hardcoded path later
    captions_paths = [
        path.replace(
            "/video_320px_15sec/",
            f"/video_320px_15sec/lavila_captions_num_frames_{args.num_frames}/",
        )
        for path in video_paths
    ]

    return video_paths, captions_paths


def custom_collate_fn(batch):
    # batch: list of dicts from your dataset __getitem__
    # {
    #   "video_path": str,
    #   "frames": Tensor [S, T, C, H, W],
    #   "frames_ids": List[int] (or Tensor),
    #   "fps": float/int,
    #   "caption_path": str,
    # }

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

    wandb.init(project="Thesis", id=args.wandb_run_name, config=args, resume="allow")

    model, tokenizer = load_model_and_tokenizer(args)

    val_transform = load_val_transform(args)

    dataset = VideoNarratorDataset(
        video_root=args.video_path_root,
        caption_suffix=f"lavila_captions_num_frames_{args.num_frames}",
        num_frames=args.num_frames,
        num_segments=args.num_segments,
        val_transform=val_transform,
        jitter=False,
    )

    print(f"len(dataset) = {len(dataset)}")

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
    )

    print(f"len(dataloader) = {len(dataloader)}")

    for sample in dataloader:
        frames = sample["frames"]
        frames_ids = sample["frames_ids"]
        fps = sample["fps"]
        video_path = sample["video_path"]
        caption_path = sample["caption_path"]

        with torch.no_grad():
            frames = frames.cuda(non_blocking=True)
            image_features = model.encode_image(frames)
            print(image_features.shape)

        break

    exit()

    # for video_path, captions_path in zip(video_paths, captions_paths):
    #     # Check if captions path exists
    #     if os.path.exists(captions_path):
    #         continue

    #     # Create a directory for the video, which is a 15 second clip
    #     os.makedirs(captions_path, exist_ok=True)

    #     frames_chunked, frame_ids_chunked, fps = load_frames(
    #         video_path=video_path,
    #         num_segments=args.num_segments,
    #         num_frames=args.num_frames,
    #         val_transform=val_transform,
    #         jitter=False,
    #     )

    #     results = []

    #     with torch.no_grad():
    #         for i, frames_chunk in enumerate(frames_chunked):
    #             frames_chunk = frames_chunk.cuda(non_blocking=True)

    #             image_features = model.encode_image(frames_chunk)

    #             generated_text_ids, _ = model.generate(
    #                 image_features,
    #                 tokenizer,
    #                 target=None,
    #                 max_text_length=args.max_text_length,
    #                 top_k=args.top_k,
    #                 top_p=args.top_p,
    #                 num_return_sequences=args.num_return_sequences,
    #                 temperature=args.temperature,
    #                 early_stopping=args.early_stopping,
    #             )

    #             generated_text_strs = generate_text(
    #                 generated_text_ids, tokenizer, args.num_return_sequences
    #             )

    #             results.append(
    #                 {
    #                     "chunk_id": i,
    #                     "generated_text_strs": generated_text_strs,
    #                     "frame_ids": frame_ids_chunked[i].tolist(),
    #                     "timestamps": [
    #                         frame_id / fps for frame_id in frame_ids_chunked[i].tolist()
    #                     ],  # Convert frame ids to timestamps
    #                     "fps": fps,
    #                 }
    #             )

    #     with open(f"{captions_path}/captions.json", "w") as f:
    #         json.dump(results, f)

    #     wandb.log({"total_videos": len(video_paths)})


def get_args_parser():
    parser = argparse.ArgumentParser(description="LAVILA narrator", add_help=True)
    parser.add_argument("--wandb-project-name", default="Thesis", type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str, required=True)
    parser.add_argument("--wandb-log-video", action="store_true")

    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=4, type=int)

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
            wandb.finish(exit_code=1)
        raise
    finally:
        if dist_utils.is_main_process():
            wandb.finish()
