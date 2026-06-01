import logging
import sys
from contextlib import nullcontext
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F


LOGGER = logging.getLogger(__name__)


def repo_root():
    return Path(__file__).resolve().parents[2]


def prepare_decord_path():
    decord_path = repo_root() / "third_party" / "decord" / "python"
    if decord_path.exists() and str(decord_path) not in sys.path:
        sys.path.insert(0, str(decord_path))


def choose_device(name):
    if name.startswith("cuda") and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but not available; falling back to CPU")
        return torch.device("cpu")
    return torch.device(name)


def strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if next(iter(state_dict)).startswith("module."):
        return OrderedDict((key.replace("module.", "", 1), value) for key, value in state_dict.items())
    return state_dict


def normalize_features(features):
    return features / features.norm(dim=-1, keepdim=True).clamp(min=1e-8)


def precision_to_dtype(precision):
    precision = str(precision or "fp16").lower()
    if precision in ("none", "off", "fp32", "float32"):
        return None
    if precision in ("auto", "amp", "fp16", "float16", "half"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError("inference_precision must be one of auto/amp/fp16/bf16/fp32/off")


def resolve_checkpoint_path(checkpoint_path, model_name):
    path = Path(checkpoint_path)
    if path.is_dir():
        for filename in ("checkpoint_best.pt", "checkpoint.pt"):
            candidate = path / filename
            if candidate.exists():
                LOGGER.info("resolved %s checkpoint directory %s -> %s", model_name, path, candidate)
                return candidate
        raise FileNotFoundError(
            f"{model_name} checkpoint path is a directory but contains neither checkpoint_best.pt nor checkpoint.pt: {path}"
        )
    return path


def load_video_segment(video_root, video_uid, start, end, cfg, clip_length):
    prepare_decord_path()
    from avion.data.clip_dataset import video_loader

    chunk_len = int(cfg.get("video_chunk_length", 15))
    fps = float(cfg.get("video_fps", 30))
    root = str(video_root)
    if chunk_len == -1:
        expected = Path(root) / f"{video_uid}.mp4"
    else:
        expected = Path(root) / f"{video_uid}.mp4"
    if not expected.exists():
        raise FileNotFoundError(f"missing video {expected}")

    return video_loader(
        root,
        video_uid,
        "mp4",
        float(start),
        float(end),
        chunk_len=chunk_len,
        fps=fps,
        clip_length=int(clip_length),
        threads=int(cfg.get("decode_threads", 1)),
        jitter=False,
    )


class LaViLaScorer:
    model_name = "lavila"
    display_name = "LaViLa"

    def __init__(self, checkpoint_path, config):
        if not checkpoint_path:
            raise FileNotFoundError(f"missing {self.display_name} checkpoint: {checkpoint_path}")
        checkpoint_path = resolve_checkpoint_path(checkpoint_path, self.display_name)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"missing {self.display_name} checkpoint: {checkpoint_path}")
        self.config = config
        self.scoring_cfg = config["scoring"]
        self.video_root = config["paths"].get("video_root")
        if not self.video_root:
            raise ValueError("video_root must be provided in config paths or --video-root")
        self.device = choose_device(self.scoring_cfg.get("device", "cuda"))
        self.inference_dtype = precision_to_dtype(self.scoring_cfg.get("inference_precision", "fp16"))
        self.checkpoint_path = str(checkpoint_path)
        self.text_cache = {}
        LOGGER.info(
            "initializing %s scorer device=%s precision=%s video_root=%s",
            self.display_name,
            self.device,
            self.inference_dtype or "fp32",
            self.video_root,
        )
        self.model, self.context_length, self.clip_length, self.crop_size = self._load_model()
        self.model.eval()
        LOGGER.info(
            "%s scorer ready context_length=%d clip_length=%d crop_size=%d",
            self.display_name,
            self.context_length,
            self.clip_length,
            self.crop_size,
        )

    def _load_model(self):
        import avion.models.model_clip as model_clip
        from avion.models.utils import inflate_positional_embeds

        LOGGER.info("loading %s checkpoint file %s", self.display_name, self.checkpoint_path)
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        old_args = ckpt.get("args")
        model_name = getattr(old_args, "model", "CLIP_VITB16")
        clip_length = (
            self.scoring_cfg.get(f"{self.model_name}_clip_length")
            or self.scoring_cfg.get(f"{self.model_name}_num_frames")
            or self.scoring_cfg.get("lavila_clip_length")
            or getattr(old_args, "clip_length", 4)
        )
        context_length = getattr(old_args, "context_length", 77)
        crop_size = 336 if str(model_name).endswith("_336PX") else 224
        model = getattr(model_clip, model_name)(
            freeze_temperature=True,
            use_grad_checkpointing=False,
            context_length=context_length,
            vocab_size=getattr(old_args, "vocab_size", 49408),
            patch_dropout=0.0,
            num_frames=clip_length,
            drop_path_rate=0.0,
            use_fast_conv1=getattr(old_args, "use_fast_conv1", False),
            use_flash_attn=getattr(old_args, "use_flash_attn", False),
            use_quick_gelu=True,
            project_embed_dim=getattr(old_args, "project_embed_dim", 256),
            pretrain_zoo=getattr(old_args, "pretrain_zoo", "openai"),
            pretrain_path=getattr(old_args, "pretrain_path", None),
        )
        state_dict = strip_module_prefix(ckpt.get("state_dict", ckpt))
        state_dict = inflate_positional_embeds(
            model.state_dict(),
            state_dict,
            num_frames=clip_length,
            load_temporal_fix="bilinear",
        )
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            LOGGER.warning("Strict %s checkpoint load failed (%s); retrying strict=False", self.display_name, exc)
            model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        return model, context_length, int(clip_length), crop_size

    def autocast_context(self):
        if self.device.type != "cuda" or self.inference_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.inference_dtype)

    def preprocess_video(self, frames):
        norm_style = self.scoring_cfg.get(f"{self.model_name}_norm_style") or self.scoring_cfg.get("lavila_norm_style", "openai")
        if norm_style == "openai":
            mean = torch.tensor([108.3272985, 116.7460125, 104.09373615000001])
            std = torch.tensor([68.5005327, 66.6321579, 70.32316305])
        elif norm_style == "timm":
            mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255])
            std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255])
        else:
            raise ValueError(f"{self.model_name}_norm_style must be openai or timm")

        video = frames.float().permute(0, 3, 1, 2)
        video = F.interpolate(video, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        video = video.permute(1, 0, 2, 3)
        mean = mean.view(3, 1, 1, 1)
        std = std.view(3, 1, 1, 1)
        return (video - mean) / std

    def encode_texts(self, texts):
        from avion.data.tokenizer import tokenize

        uncached = [text for text in texts if text not in self.text_cache]
        if uncached:
            tokens = tokenize(uncached, context_length=self.context_length).to(self.device)
            with torch.no_grad(), self.autocast_context():
                features = self.model.encode_text(tokens)
            features = normalize_features(features.float())
            for text, feature in zip(uncached, features):
                self.text_cache[text] = feature.detach()
        return torch.stack([self.text_cache[text] for text in texts], dim=0)

    def score_record(self, record):
        return score_record_with_scorer(record, self)

    def encode_videos(self, video_uid, candidates):
        videos = []
        good_indices = []
        for idx, candidate in enumerate(candidates):
            try:
                frames = load_video_segment(
                    self.video_root,
                    video_uid,
                    candidate["start_sec"],
                    candidate["end_sec"],
                    self.scoring_cfg,
                    self.clip_length,
                )
                videos.append(self.preprocess_video(frames))
                good_indices.append(idx)
            except Exception as exc:
                candidate.setdefault("scoring_errors", []).append(f"{self.model_name}_decoding_failure: {exc}")
        if not videos:
            return [], None
        return good_indices, torch.stack(videos, dim=0).to(self.device)

    def compute_video_features(self, videos):
        with torch.no_grad(), self.autocast_context():
            features = self.model.encode_image(videos)
        return normalize_features(features.float())


class EgoVLPScorer(LaViLaScorer):
    model_name = "egovlp"
    display_name = "EgoVLP"


def text_slots(record):
    slots = [("current", record.get("text"))]
    if record.get("previous_text"):
        slots.append(("prev", record.get("previous_text")))
    if record.get("next_text"):
        slots.append(("next", record.get("next_text")))
    return [(name, text) for name, text in slots if isinstance(text, str) and text.strip()]


def score_record_with_scorer(record, scorer):
    candidates = record.get("candidates") or []
    if not candidates:
        record.setdefault("errors", []).append("too_few_valid_candidates")
        return record
    slots = text_slots(record)
    if not slots or slots[0][0] != "current":
        record.setdefault("errors", []).append("missing_text")
        return record

    labels = [name for name, _ in slots]
    texts = [text for _, text in slots]
    text_features = scorer.encode_texts(texts)

    batch_size = int(scorer.scoring_cfg.get("batch_size", 16))
    video_uid = record.get("video_uid")
    for start_idx in range(0, len(candidates), batch_size):
        batch_candidates = candidates[start_idx : start_idx + batch_size]
        good_indices, videos = scorer.encode_videos(video_uid, batch_candidates)
        if videos is None:
            continue
        video_features = scorer.compute_video_features(videos)
        sims = video_features @ text_features.t()
        sims = sims.detach().cpu().float()
        for row_idx, local_idx in enumerate(good_indices):
            candidate = batch_candidates[local_idx]
            for col_idx, label in enumerate(labels):
                candidate[f"{scorer.model_name}_sim_{label}"] = float(sims[row_idx, col_idx].item())
    scored_any = any(f"{scorer.model_name}_sim_current" in candidate for candidate in candidates)
    if not scored_any:
        record.setdefault("errors", []).append(f"{scorer.model_name}_all_candidate_decoding_failed")
        LOGGER.warning(
            "all candidate decoding/scoring failed model=%s narration_uid=%s video_uid=%s candidates=%d",
            scorer.model_name,
            record.get("narration_uid"),
            video_uid,
            len(candidates),
        )
    return record
