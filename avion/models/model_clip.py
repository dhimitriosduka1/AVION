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


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        vision_model: nn.Module,
        text_model: nn.Module,
        vision_width: int = None,
        text_width: int = None,
        freeze_temperature=False,
        tau_min=0.04,
        tau_max=0.12,
        anchor_span=1.0,
        alpha=0.25,
        enable_temperature_modulation=True,
        **kwargs,
    ):
        super().__init__()

        self.visual = vision_model
        self.textual = text_model

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        if freeze_temperature or enable_temperature_modulation:
            self.logit_scale.requires_grad_(False)

        if vision_width is not None:
            self.vision_width = vision_width
            self.image_projection = nn.Parameter(torch.empty(vision_width, embed_dim))
        else:
            self.image_projection = None
        if text_width is not None:
            self.text_width = text_width
            self.text_projection = nn.Parameter(torch.empty(text_width, embed_dim))

        self.enable_temperature_modulation = enable_temperature_modulation
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.anchor_span = anchor_span
        self.alpha = alpha

        self.init_parameters()

    def compute_per_pair_logit_scale(self, clip_length):
        factor = torch.pow(clip_length / self.anchor_span, self.alpha)
        modulated_tau = self.base_temperature * factor
        
        modulated_tau = torch.clamp(modulated_tau, self.tau_min, self.tau_max)
        return 1.0 / modulated_tau

    def forward(self, image, text, clip_length=None):
        image_embed = F.normalize(self.encode_image(image), dim=-1)
        text_embed = F.normalize(self.encode_text(text), dim=-1)

        if self.enable_temperature_modulation and clip_length is not None:
            logit_scale = self.compute_per_pair_logit_scale(clip_length)
        else:
            logit_scale = self.logit_scale.exp()

        return image_embed, text_embed, logit_scale

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
