import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LoRAConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: Tuple[str, ...] = ()


class LoRALinear(nn.Module):
    """
    Minimal LoRA wrapper for nn.Linear.
    - Base weights are frozen.
    - Only LoRA A/B are trainable.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # A: down-proj, B: up-proj (LoRA paper convention)
        self.lora_A = nn.Parameter(torch.zeros((self.r, base.in_features), dtype=base.weight.dtype, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros((base.out_features, self.r), dtype=base.weight.dtype, device=base.weight.device))

        # Mark trainability for tools.RequiresGrad (see tools.py patch).
        self.lora_A._trainable = True
        self.lora_B._trainable = True
        self.base.weight._trainable = False
        if self.base.bias is not None:
            self.base.bias._trainable = False

        # Init
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # LoRA: x @ A^T @ B^T * scaling
        lora = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scaling
        return out + lora


class LoRAMultiheadAttention(nn.Module):
    """
    LoRA wrapper for torch.nn.MultiheadAttention in MineCLIP's CLIP implementation.

    MineCLIP uses nn.MultiheadAttention with a *combined* in_proj_weight (qkv).
    We add a low-rank update to in_proj_weight and to out_proj.weight.

    Note: For simplicity, this implements LoRA as a weight-delta on the projection matrices.
    """

    def __init__(self, base: nn.MultiheadAttention, r: int, alpha: float, dropout: float):
        super().__init__()
        if not isinstance(base, nn.MultiheadAttention):
            raise TypeError(f"LoRAMultiheadAttention expects nn.MultiheadAttention, got {type(base)}")
        if r <= 0:
            raise ValueError("LoRA rank r must be > 0")

        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.r
        self.dropout = float(dropout)

        E = base.embed_dim
        # in_proj_weight: (3E, E)
        self.in_A = nn.Parameter(torch.zeros((self.r, E), dtype=base.in_proj_weight.dtype, device=base.in_proj_weight.device))
        self.in_B = nn.Parameter(torch.zeros((3 * E, self.r), dtype=base.in_proj_weight.dtype, device=base.in_proj_weight.device))
        nn.init.kaiming_uniform_(self.in_A, a=5**0.5)
        nn.init.zeros_(self.in_B)

        # out_proj: Linear(E -> E)
        self.out_A = nn.Parameter(torch.zeros((self.r, E), dtype=base.out_proj.weight.dtype, device=base.out_proj.weight.device))
        self.out_B = nn.Parameter(torch.zeros((E, self.r), dtype=base.out_proj.weight.dtype, device=base.out_proj.weight.device))
        nn.init.kaiming_uniform_(self.out_A, a=5**0.5)
        nn.init.zeros_(self.out_B)

        # Freeze base params
        for p in self.base.parameters():
            p.requires_grad_(False)
            p._trainable = False
        self.in_A._trainable = True
        self.in_B._trainable = True
        self.out_A._trainable = True
        self.out_B._trainable = True

    def _delta_in(self) -> torch.Tensor:
        return (self.in_B @ self.in_A) * self.scaling

    def _delta_out(self) -> torch.Tensor:
        return (self.out_B @ self.out_A) * self.scaling

    def forward(self, query, key, value, **kwargs):
        # Mirror nn.MultiheadAttention forward by directly calling functional MHA with injected weights.
        # Keep signature flexible since MineCLIP calls it as self.attn(x,x,x, need_weights=False, attn_mask=...).
        need_weights = kwargs.pop("need_weights", True)
        attn_mask = kwargs.pop("attn_mask", None)
        key_padding_mask = kwargs.pop("key_padding_mask", None)
        average_attn_weights = kwargs.pop("average_attn_weights", True)
        is_causal = kwargs.pop("is_causal", False)
        if kwargs:
            raise TypeError(f"Unexpected kwargs for LoRAMultiheadAttention: {list(kwargs.keys())}")

        in_w = self.base.in_proj_weight + self._delta_in()
        out_w = self.base.out_proj.weight + self._delta_out()

        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.base.embed_dim,
            num_heads=self.base.num_heads,
            in_proj_weight=in_w,
            in_proj_bias=self.base.in_proj_bias,
            bias_k=self.base.bias_k,
            bias_v=self.base.bias_v,
            add_zero_attn=self.base.add_zero_attn,
            dropout_p=self.base.dropout,
            out_proj_weight=out_w,
            out_proj_bias=self.base.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            is_causal=is_causal,
        )


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    for p in patterns:
        if re.search(p, name):
            return True
    return False


def inject_lora(
    model: nn.Module,
    *,
    r: int,
    alpha: float,
    dropout: float,
    target_patterns: List[str],
    verbose: bool = False,
) -> int:
    """
    In-place LoRA injection into:
    - nn.Linear modules (replaced with LoRALinear) when name matches target_patterns
    - nn.MultiheadAttention modules (replaced with LoRAMultiheadAttention) when name matches target_patterns

    Returns number of modules replaced.
    """
    replaced = 0
    patterns = list(target_patterns or [])

    # Replace by walking named_modules and patching on parent.
    for full_name, module in list(model.named_modules()):
        if full_name == "":
            continue
        if not _matches_any(full_name, patterns):
            continue

        # Find parent
        parent = model
        parts = full_name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]

        child = getattr(parent, leaf)
        if isinstance(child, nn.Linear):
            setattr(parent, leaf, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
            if verbose:
                print(f"[LoRA] Replaced Linear: {full_name}")
        elif isinstance(child, nn.MultiheadAttention):
            setattr(parent, leaf, LoRAMultiheadAttention(child, r=r, alpha=alpha, dropout=dropout))
            replaced += 1
            if verbose:
                print(f"[LoRA] Replaced MultiheadAttention: {full_name}")

    return replaced


def freeze_module_params(module: nn.Module, *, trainable: bool = False):
    """
    Freeze/unfreeze module parameters and mark trainability for tools.RequiresGrad.
    """
    for p in module.parameters():
        p.requires_grad_(trainable)
        p._trainable = bool(trainable)

