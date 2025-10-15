import torch
import argparse
import open_clip
from tqdm import tqdm
from data.datasets import VideoMetadataDataset


def get_args_parser():
    parser = argparse.ArgumentParser()
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
    return model, tokenizer


@torch.no_grad()
@torch.autocast("cuda")
def encode_text(model, text):
    # The text is already tokenized
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.pretrained, device
    )

    # Load video metadata dataset
    video_metadata_dataset = VideoMetadataDataset(args.video_metadata_path, tokenizer)

    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(
        video_metadata_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    for batch in tqdm(dataloader, desc="Encoding text"):
        caption, frequency = batch
        text_features = encode_text(model, caption)
        print(text_features)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
