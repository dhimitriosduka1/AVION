import os
import wandb
import argparse
import urllib.request
import avion.utils.distributed as dist_utils
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

from lavila.data.video_transforms import Permute
from lavila.data.datasets import get_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from lavila.models.tokenizer import MyGPT2Tokenizer


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


def main(args):

    wandb.init(project="Thesis", id=args.wandb_run_name, config=args, resume="allow")

    frames = get_frames(
        args.video_path_root, args.video_path, args.num_segments, jitter=False
    )

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

    # transforms on input frames
    val_transform = transforms.Compose(
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

    frames = val_transform(frames)
    frames = frames.unsqueeze(0)

    tokenizer = MyGPT2Tokenizer(args.tokenizer_name, add_bos=True)

    with torch.no_grad():
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

    generated_text_strs = []
    for i in range(args.num_return_sequences):
        generated_text_str = decode_one(generated_text_ids[i], tokenizer)
        generated_text_strs.append(generated_text_str)
        print("{}: {}".format(i, generated_text_str))

    # Log the generated text strings and video on wandb
    video_path = os.path.join(args.video_path_root, args.video_path)
    video_data = wandb.Video(video_path, caption="Generated Video")
    wandb.log({"generated_text_strs": generated_text_strs, "video": video_data})


def get_args_parser():
    parser = argparse.ArgumentParser(description="LAVILA narrator", add_help=True)
    parser.add_argument("--wandb-project-name", default="Thesis", type=str)
    parser.add_argument("--wandb-run-name", default=None, type=str, required=True)

    parser.add_argument(
        "--video-path-root",
        default="/ptmp/dduka/databases/ego4d/video_320px_15sec",
        type=str,
    )

    # To remove
    parser.add_argument(
        "--video-path",
        default="eaa3ba4c-0da9-47c8-9aca-0b5cc83c902a.mp4/0.mp4",
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
