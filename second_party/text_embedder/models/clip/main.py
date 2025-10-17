import os
import wandb
import torch
import argparse
import open_clip

from tqdm import tqdm
from safetensors.torch import save_file

from second_party.text_embedder.data.datasets import VideoMetadataDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--video-metadata-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser


def load_model_and_tokenizer(model_name, pretrained, device):
    model, _, __ = open_clip.create_model_and_transforms(model_name, pretrained)
    model.eval()
    model.to(device)

    tokenizer = open_clip.get_tokenizer(model_name)

    with torch.no_grad():
        dim = model.encode_text(tokenizer(["foo"]).to(device)).shape[-1]
        print(f"Dimension of the text embeddings: {dim}")

    return model, tokenizer, dim


@torch.no_grad()
@torch.autocast("cuda")
def encode_text(model, text, normalize=True):
    # The text is already tokenized
    return model.encode_text(text, normalize=normalize)


def main(args):
    wandb.init(
        project="Thesis",
        name=f"{args.model_name}_{args.pretrained}_{args.video_metadata_path.split('/')[-2]}",
        config={**args.__dict__},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer, dim = load_model_and_tokenizer(
        args.model_name, args.pretrained, device
    )

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

    embeddings = {}
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Encoding text")):
        original_caption, caption, _ = (
            batch["original_caption"],
            batch["caption"],
            batch["frequency"],
        )

        caption = caption.to(device)

        text_features = encode_text(model, caption).detach().float().cpu()

        for i in range(len(original_caption)):
            embeddings[original_caption[i]] = text_features[i]

        wandb.log(
            {
                "progress": (batch_idx + 1) / len(dataloader),
            }
        )

    print(f"Saving embeddings")
    save_file(
        embeddings,
        os.path.join(output_dir, f"embeddings.safetensors"),
        metadata={
            "model": args.model_name,
            "pretrained": args.pretrained,
            "video_metadata_path": args.video_metadata_path,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
    )

    print(f"Done")


# Run the script using: python3 -m second_party.text_embedder.models.clip.main + args
if __name__ == "__main__":
    args = get_args_parser().parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        wandb.finish(exit_code=1)
    finally:
        wandb.finish()
