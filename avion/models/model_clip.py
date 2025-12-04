import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from avion.models.transformer import TextTransformer, VisionTransformer
from avion.models.utils import (
    enable_grad_checkpointing,
    remap_keys_from_open_clip_to_vit,
)

from transformers import AutoModel


class VideoClassifier(nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        dropout: float,
        num_classes: int,
        **kwargs,
    ):
        super().__init__()
        self.visual = vision_model
        self.dropout = nn.Dropout(dropout)
        if hasattr(self.visual, "image_projection"):
            self.visual.image_projection = None
        self.fc_cls = nn.Linear(vision_model.width, num_classes, bias=True)
        self.fc_cls.weight.data.normal_(mean=0.0, std=0.01)
        self.fc_cls.bias.data.zero_()

    def forward(self, image):
        image_embed = self.visual(image)
        if isinstance(image_embed, list):
            assert len(image_embed) == 1
            image_embed = image_embed[0]
        logit = self.fc_cls(self.dropout(image_embed))
        return logit


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vision_model: nn.Module,
        text_model: nn.Module,
        vision_width: int = None,
        text_width: int = None,
        freeze_temperature=False,
        **kwargs,
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if freeze_temperature:
            self.logit_scale.requires_grad_(False)
        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))
        else:
            self.text_projection = None

        self.init_parameters()

    def init_parameters(self):
        if self.image_projection is not None:
            trunc_normal_(self.image_projection, std=self.vision_width**-0.5)
        if self.text_projection is not None:
            trunc_normal_(self.text_projection, std=self.text_width**-0.5)

    def encode_image(self, image):
        x = self.visual(image)
        if self.image_projection is not None:
            x = x @ self.image_projection.to(x.dtype)
        return x

    def encode_text(self, text, cast_dtype=None):
        x = self.textual(text, cast_dtype=cast_dtype)
        if self.text_projection is not None:
            x = x @ self.text_projection.to(x.dtype)
        return x

    def forward(self, image, text):
        image_embed = self.encode_image(image)
        text_embed = self.encode_text(text, cast_dtype=image_embed.dtype)

        return (
            F.normalize(image_embed, dim=-1),
            F.normalize(text_embed, dim=-1),
            self.logit_scale.exp(),
        )


def CLIP_VITB16(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    patch_dropout=0.0,
    drop_path_rate=0.0,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo="openai",
    pretrain_path=None,
    **kwargs,
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224,
        16,
        768,
        12,
        12,
        4,
        output_dim=project_embed_dim,
        patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(
        context_length=context_length,
        vocab_size=49408,
        width=512,
        heads=8,
        layers=12,
        output_dim=project_embed_dim,
        causal_mask=not use_bidirectional_lm,
    )
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_model=vision_model,
        text_model=text_model,
        freeze_temperature=freeze_temperature,
    )

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load("ViT-B/16", device="cpu")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_state_dict, strict=False
        )
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            state_dict, use_fast_conv1=use_fast_conv1, use_flash_attn=use_flash_attn
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_state_dict, strict=False
        )
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_VITL14(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.0,
    drop_path_rate=0.0,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo="openai",
    pretrain_path=None,
    **kwargs,
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        224,
        14,
        1024,
        24,
        16,
        4,
        output_dim=project_embed_dim,
        patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(
        context_length=context_length,
        vocab_size=vocab_size,
        width=768,
        heads=12,
        layers=12,
        output_dim=project_embed_dim,
        causal_mask=not use_bidirectional_lm,
    )
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_model=vision_model,
        text_model=text_model,
        freeze_temperature=freeze_temperature,
    )

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load("ViT-L/14", device="cpu")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            24,
            context_length=context_length,
            vocab_size=vocab_size,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_state_dict, strict=False
        )
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            state_dict, use_flash_attn=use_flash_attn
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_state_dict, strict=False
        )
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


def CLIP_VITL14_336PX(
    freeze_temperature=False,
    use_grad_checkpointing=False,
    use_bidirectional_lm=False,
    context_length=77,
    vocab_size=49408,
    patch_dropout=0.0,
    drop_path_rate=0.0,
    num_frames=1,
    use_fast_conv1=False,
    use_flash_attn=False,
    project_embed_dim=512,
    pretrain_zoo="openai",
    pretrain_path=None,
    **kwargs,
):
    # vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    vision_model = VisionTransformer(
        336,
        14,
        1024,
        24,
        16,
        4,
        output_dim=project_embed_dim,
        patch_dropout=patch_dropout,
        drop_path_rate=drop_path_rate,
        num_frames=num_frames,
        use_fast_conv1=use_fast_conv1,
        use_flash_attn=use_flash_attn,
    )
    text_model = TextTransformer(
        context_length=context_length,
        vocab_size=vocab_size,
        width=768,
        heads=12,
        layers=12,
        output_dim=project_embed_dim,
        causal_mask=not use_bidirectional_lm,
    )
    enable_grad_checkpointing(vision_model, use_grad_checkpointing)
    enable_grad_checkpointing(text_model, use_grad_checkpointing)
    model = CLIP(
        embed_dim=project_embed_dim,
        vision_model=vision_model,
        text_model=text_model,
        freeze_temperature=freeze_temperature,
    )

    if pretrain_zoo == "openai":
        print("=> loading openai model")
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            clip_model.state_dict(),
            24,
            context_length=context_length,
            vocab_size=vocab_size,
            use_fast_conv1=use_fast_conv1,
            use_flash_attn=use_flash_attn,
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_state_dict, strict=False
        )
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    elif pretrain_zoo == "open_clip":
        assert pretrain_path is not None
        state_dict = torch.load(pretrain_path)
        print("=> loading open_clip model")
        remapped_state_dict = remap_keys_from_open_clip_to_vit(
            state_dict, use_flash_attn=use_flash_attn
        )
        missing_keys, unexpected_keys = model.load_state_dict(
            remapped_state_dict, strict=False
        )
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)
    else:
        raise NotImplementedError
    return model


class VJEPAAdapter(nn.Module):
    def __init__(self, vjepa_model):
        super().__init__()
        self.model = vjepa_model

    def forward(self, x):
        # x input shape: (B, C, T, H, W)
        # HF V-JEPA expects: (B, T, C, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Forward pass
        # Note: If your V-JEPA model requires pixel_values explicitly, use:
        # outputs = self.model(pixel_values=x)
        outputs = self.model(x)

        # Get Last Hidden State: (B, Sequence_Length, Hidden_Dim)
        features = outputs.last_hidden_state

        # Global Average Pooling (GAP)
        # Averages over the sequence dimension (T * H * W patches)
        # Result: (B, Hidden_Dim)
        return features.mean(dim=1)


def VJEPA2_VITL256(
    freeze_backbone=False,
    freeze_temperature=False,
    context_length=77,
    vocab_size=49408,
    project_embed_dim=512,
    vision_width=1024,
    use_bidirectional_lm=False,
    **kwargs,
):
    vjepa_model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256")

    if hasattr(vjepa_model, "gradient_checkpointing_enable"):
        print("=> Enabling Non-Reentrant Gradient Checkpointing for V-JEPA")
        vjepa_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        print(
            "Warning: V-JEPA model does not support native gradient checkpointing enable."
        )

    if freeze_backbone:
        print("=> Mode: Linear Probing (Backbone Frozen)")
        for param in vjepa_model.parameters():
            param.requires_grad = False
    else:
        print("=> Mode: Full Fine-Tuning (Backbone Unfrozen)")
        for param in vjepa_model.parameters():
            param.requires_grad = True

    vision_model = VJEPAAdapter(vjepa_model)

    text_model = TextTransformer(
        context_length=context_length,
        vocab_size=vocab_size,
        width=768,
        heads=12,
        layers=12,
        output_dim=project_embed_dim,
        causal_mask=not use_bidirectional_lm,
    )

    if use_grad_checkpointing:
        enable_grad_checkpointing(text_model, use_grad_checkpointing)

    model = CLIP(
        embed_dim=project_embed_dim,
        vision_model=vision_model,
        text_model=text_model,
        vision_width=vision_width,
        text_width=None,
        freeze_temperature=freeze_temperature,
    )

    if pretrain_zoo == "openai":
        print("=> Loading OpenAI CLIP Text weights (ViT-L/14)")
        try:
            clip_model, _ = clip.load("ViT-L/14", device="cpu")
            remapped_state_dict = remap_keys_from_open_clip_to_vit(
                clip_model.state_dict(),
                use_fast_conv1=kwargs.get("use_fast_conv1", False),
                use_flash_attn=kwargs.get("use_flash_attn", False),
            )

            text_dict = {
                k: v for k, v in remapped_state_dict.items() if "visual" not in k
            }

            model.load_state_dict(text_dict, strict=False)
        except Exception as e:
            print(f"Warning: Could not load OpenAI text weights: {e}")

    return model
