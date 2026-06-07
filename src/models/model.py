"""
model.py
========
Model construction, LoRA implementation, and fine-tuning strategies
for DSA-LoRA: Domain-Shift-Aware Parameter-Efficient Fine-Tuning
for Malaria Detection in Low-Resource Clinical Settings.
"""

import torch
import torch.nn as nn
import timm


# ── MODEL CONSTRUCTION ────────────────────────────────────────────────────────

def build_vit_model(num_classes=2, pretrained=True):
    """
    Load ViT-B/16 pretrained on ImageNet.
    Replace the classification head for binary malaria detection.

    Args:
        num_classes: number of output classes (2 for binary detection)
        pretrained:  whether to load ImageNet pretrained weights

    Returns:
        model: ViT-B/16 model ready for fine-tuning
    """
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=pretrained,
        num_classes=num_classes
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ViT-B/16 | Total: {total:,} | Trainable: {trainable:,}")
    return model


# ── LORA IMPLEMENTATION ───────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) wrapper for a linear layer.

    Adds trainable low-rank matrices A and B alongside the frozen
    original weight matrix W, such that:

        output = W*x + scale * (B @ A) * x
        where scale = alpha / rank

    Only A and B are trained. W is frozen throughout fine-tuning.
    This reduces trainable parameters from (out * in) to (rank * in + out * rank).

    Args:
        linear_layer: the original nn.Linear layer to wrap
        rank:         rank of the low-rank decomposition (default: 4)
        alpha:        scaling factor for the LoRA update (default: 8)
    """
    def __init__(self, linear_layer, rank=4, alpha=8):
        super().__init__()
        in_features  = linear_layer.in_features
        out_features = linear_layer.out_features

        self.linear = linear_layer       # frozen original weights
        self.scale  = alpha / rank       # scaling factor

        # LoRA matrices — small and trainable
        # A initialised with small random values, B initialised to zero
        # so the LoRA update is zero at the start of fine-tuning
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scale


# ── FINE-TUNING STRATEGIES ────────────────────────────────────────────────────

def apply_lora_to_layers(model, target_layers, rank=4, alpha=8):
    """
    Apply LoRA adapters to Q/V projection matrices in specified
    transformer blocks. All other parameters are frozen.

    This is the core of DSA-LoRA: by passing only the high-MMD layers
    as target_layers, adaptation is concentrated where domain shift
    is largest, preserving stable representations elsewhere.

    Args:
        model:         ViT-B/16 model (loaded from Stage 1 checkpoint)
        target_layers: list of block indices to apply LoRA to
                       e.g. [4, 7, 8, 9, 10, 11] from MMD analysis
        rank:          LoRA rank (default: 4)
        alpha:         LoRA scaling factor (default: 8)

    Returns:
        model:            model with LoRA applied
        trainable_params: number of trainable parameters
        total_params:     total number of parameters
    """
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Always keep classification head trainable
    for param in model.head.parameters():
        param.requires_grad = True

    # Apply LoRA to specified transformer blocks
    for block_idx in target_layers:
        block = model.blocks[block_idx]
        attn  = block.attn

        # Replace fused QKV projection with LoRA-wrapped version
        # timm ViT uses a single qkv linear: [3*hidden_dim, hidden_dim]
        if hasattr(attn, 'qkv'):
            attn.qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha)

        # Keep attention output projection trainable
        if hasattr(attn, 'proj'):
            for param in attn.proj.parameters():
                param.requires_grad = True

        # Keep layer norms trainable — important for distribution shift
        for param in block.norm1.parameters():
            param.requires_grad = True
        for param in block.norm2.parameters():
            param.requires_grad = True

    # Final layer norm
    for param in model.norm.parameters():
        param.requires_grad = True

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    pct = 100 * trainable_params / total_params

    print(f"LoRA applied to layers {target_layers}")
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({pct:.2f}%)")

    return model, trainable_params, total_params


def apply_standard_lora(model, rank=4, alpha=8):
    """
    Apply LoRA uniformly to ALL 12 transformer blocks.
    Used as the Standard LoRA baseline in Stage 3 experiments.

    Args:
        model: ViT-B/16 model
        rank:  LoRA rank
        alpha: LoRA scaling factor

    Returns:
        model, trainable_params, total_params
    """
    all_layers = list(range(12))
    return apply_lora_to_layers(model, all_layers, rank=rank, alpha=alpha)


def apply_full_finetuning(model):
    """
    Unfreeze all parameters for full fine-tuning.
    Used as the Full Fine-Tuning baseline in Stage 3 experiments.

    Args:
        model: ViT-B/16 model

    Returns:
        model:            model with all parameters trainable
        trainable_params: total parameter count
        total_params:     total parameter count (same as trainable)
    """
    for param in model.parameters():
        param.requires_grad = True

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Full fine-tuning | Trainable: {trainable:,} / {total:,} (100.00%)")
    return model, trainable, total


def load_source_model(checkpoint_path, device, num_classes=2):
    """
    Load the Stage 1 source domain model from a checkpoint.

    Args:
        checkpoint_path: path to source_domain_model_final.pth
        device:          torch device
        num_classes:     number of output classes

    Returns:
        model: loaded ViT-B/16 model in eval mode
    """
    model = timm.create_model(
        'vit_base_patch16_224',
        pretrained=False,
        num_classes=num_classes
    )
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    print(f"Source model loaded from: {checkpoint_path}")
    return model
