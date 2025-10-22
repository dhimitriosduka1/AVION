import os
import wandb
import torch
import argparse
import numpy as np
import json
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from second_party.text_embedder.data.datasets import VideoMetadataDataset


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--video-metadata-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--flush-frequency", type=int, default=200)
    return parser


def load_model_and_tokenizer(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.eval()
    model.to(device)

    sentences = ["foo"]
    with torch.no_grad():
        encoded_input = tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        )

        encoded_input = encoded_input.to(device)

        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(
            model_output, encoded_input["attention_mask"]
        )
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        dim = sentence_embeddings.shape[-1]
        print(f"Dimension of the text embeddings: {dim}")

    return model, tokenizer, dim


def make_collate_fn(tokenizer):
    def collate(samples):
        original_caps = [s["original_caption"] for s in samples]
        texts = [s["caption"] for s in samples]
        freqs = [s.get("frequency", 1) for s in samples]

        batch_enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        return {
            "original_caption": original_caps,
            "caption": batch_enc,
            "frequency": torch.as_tensor(freqs),
        }

    return collate


@torch.no_grad()
@torch.autocast("cuda")
def encode_text(model, encoded_input):
    # The text is already tokenized
    model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return F.normalize(sentence_embeddings, p=2, dim=1)


def main(args):
    wandb.init(
        project="Thesis",
        name=f"{args.model_name}_{args.video_metadata_path.split('/')[-2]}",
        config={**args.__dict__},
        group=f"Text Embeddings",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer, dim = load_model_and_tokenizer(args.model_name, device)

    print(f"Loaded model and tokenizer")

    video_metadata_dataset = VideoMetadataDataset(args.video_metadata_path, None)
    print(f"Created video metadata dataset")

    # Create the dataloader
    print(f"Creating dataloader")
    dataloader = torch.utils.data.DataLoader(
        video_metadata_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate_fn(tokenizer),
    )
    print(f"Created dataloader")

    output_dir = os.path.join(os.path.dirname(args.output_path), f"{args.model_name}")
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
        original_caption, caption, _ = (
            batch["original_caption"],
            batch["caption"],
            batch["frequency"],
        )

        caption = caption.to(device)

        text_features = encode_text(model, caption)
        text_features = text_features.detach().float().cpu().numpy()

        for i in range(len(original_caption)):
            global_index = batch_idx * args.batch_size + i
            memmap[global_index] = text_features[i]

            captions[original_caption[i]] = global_index

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


# Run the script using: python3 -m second_party.text_embedder.models.minilm.main + args
if __name__ == "__main__":
    args = get_args_parser().parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        wandb.finish(exit_code=1)
    finally:
        wandb.finish()
