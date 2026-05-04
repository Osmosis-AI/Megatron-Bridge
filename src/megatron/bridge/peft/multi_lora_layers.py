# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-adapter LoRA layer for Megatron parallel linears.

:class:`MultiLoRALinear` wraps a single Megatron parallel linear module with
*N* concurrent LoRA adapters.  The active adapter is selected at forward time
via per-layer ``tokens_per_adapter`` set by :func:`set_tokens_per_adapter_slot`.

Forward stacks the raw weights of all adapters and uses ``torch._grouped_mm``
for a single fused kernel; TP/SP collectives are issued once around the two
GEMMs to match the layout of the wrapped base linear.
"""

import math
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import (
    gather_from_sequence_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    scatter_to_sequence_parallel_region,
)

from megatron.bridge.peft.adapter_wrapper import AdapterWrapper
from megatron.bridge.peft.utils import ParallelLinearAdapter, all2all_hp2sp, get_adapter_attributes_from_linear


class SimpleLoRAAdapter(nn.Module):
    """Lightweight LoRA adapter for plain ``nn.Linear`` modules.

    Holds a ``linear_in`` (A) and ``linear_out`` (B) pair with scaling.
    Unlike :class:`ParallelLinearAdapter`, this has no TP/SP communication.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dim: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: str = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.alpha = alpha

        dtype = lora_dtype
        self.linear_in = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
        self.linear_out = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)

        self._init_weights(lora_A_init_method)

        if dropout > 0.0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = nn.Identity()
        self.dropout_position = dropout_position

    def _init_weights(self, lora_A_init_method: str) -> None:
        if lora_A_init_method == "xavier":
            nn.init.xavier_normal_(self.linear_in.weight.data)
        elif lora_A_init_method == "kaiming":
            nn.init.kaiming_uniform_(self.linear_in.weight.data, a=math.sqrt(5))
        else:
            nn.init.xavier_normal_(self.linear_in.weight.data)
        nn.init.zeros_(self.linear_out.weight.data)

    def _get_init_fn(self, init_method: str):
        if init_method == "xavier":
            return nn.init.xavier_normal_
        elif init_method == "kaiming":
            from megatron.bridge.peft.utils import init_method_kaiming_uniform
            return init_method_kaiming_uniform(math.sqrt(5))
        elif init_method == "zero":
            return lambda t: nn.init.constant_(t, 0.0)
        elif init_method == "normal":
            from megatron.bridge.peft.utils import init_method_normal
            return init_method_normal(0.2)
        raise NotImplementedError(f"Unknown init method: {init_method}")

    def forward(self, x: torch.Tensor, apply_scaling: bool = True) -> torch.Tensor:
        if self.dropout_position == "pre":
            x = self.dropout(x)
        out = self.linear_out(self.linear_in(x))
        if apply_scaling:
            out = out * (self.alpha / self.dim)
        if self.dropout_position == "post":
            out = self.dropout(out)
        return out


class SimpleMultiLoRALinear(nn.Linear):
    """Plain ``nn.Linear`` wrapped with *N* concurrent LoRA adapters.

    Extends ``nn.Linear`` (like :class:`LinearAdapter`), copies the original
    weights, freezes them, and adds N :class:`SimpleLoRAAdapter` instances.
    Returns a plain tensor — compatible with HF models.

    Args:
        orig_linear: The original ``nn.Linear`` to adapt.
        n_adapters: Number of adapter slots.
        dim: LoRA rank.
        alpha: LoRA scaling parameter.
        dropout: Dropout probability.
        dropout_position: ``'pre'`` or ``'post'``.
        lora_A_init_method: Init method for the A matrix.
        lora_dtype: Data type for adapter weights.
    """

    def __init__(
        self,
        orig_linear: nn.Linear,
        n_adapters: int,
        dim: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
        dropout_position: Literal["pre", "post"] = "pre",
        lora_A_init_method: str = "xavier",
        lora_dtype: Optional[torch.dtype] = None,
    ) -> None:
        assert isinstance(orig_linear, nn.Linear)
        super().__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)

        # Freeze base weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.n_adapters = n_adapters
        self.column_init_method = lora_A_init_method
        self.row_init_method = "zero"
        self._adapter_enabled = True
        self.tokens_per_adapter: Optional[torch.Tensor] = None
        self.max_rank = dim

        dtype = lora_dtype or orig_linear.weight.dtype
        device = orig_linear.weight.device
        self.alpha_values = torch.ones(n_adapters, dtype=dtype, device=device)
        self.rank_values = torch.full((n_adapters,), dim, dtype=dtype, device=device)
        self.adapters = nn.ModuleList([
            SimpleLoRAAdapter(
                orig_linear.in_features,
                orig_linear.out_features,
                dim=dim,
                alpha=alpha,
                dropout=dropout,
                dropout_position=dropout_position,
                lora_A_init_method=lora_A_init_method,
                lora_dtype=dtype,
                device=orig_linear.weight.device,
            )
            for _ in range(n_adapters)
        ])

    def enable_adapter_layers(self) -> None:
        self._adapter_enabled = True

    def disable_adapter_layers(self) -> None:
        self._adapter_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = torch.nn.functional.linear(x, self.weight, self.bias)

        if not self._adapter_enabled:
            return base_out

        tokens_per_adapter = self.tokens_per_adapter
        x_flat = x.reshape(-1, x.shape[-1])
        offsets = tokens_per_adapter.cumsum(dim=0)
        total = offsets[-1].item()
        assert total == x_flat.shape[0], (
            f"tokens_per_adapter sum {total} != token count {x_flat.shape[0]}"
        )

        adapter_outputs = []
        prev = 0
        for i in range(self.n_adapters):
            cur = offsets[i].item()
            if cur == prev:
                prev = cur
                continue
            out = self.adapters[i](x_flat[prev:cur], apply_scaling=False)
            adapter_outputs.append(out)
            prev = cur

        if not adapter_outputs:
            return base_out

        adapter_output = torch.cat(adapter_outputs, dim=0)

        scaling = self.alpha_values / self.rank_values
        per_token_scaling = torch.repeat_interleave(scaling, tokens_per_adapter).unsqueeze(-1)
        adapter_output = adapter_output * per_token_scaling

        return base_out + adapter_output.reshape(base_out.shape)

    # --- Per-adapter lifecycle (same interface as MultiLoRALinear) ---

    def reset_adapter(self, idx: int) -> None:
        adapter = self.adapters[idx]
        adapter._get_init_fn(self.column_init_method)(adapter.linear_in.weight.data)
        adapter._get_init_fn(self.row_init_method)(adapter.linear_out.weight.data)

    def init_adapter_slot(self, idx: int, rank: int, alpha: float) -> None:
        """Claim slot ``idx`` for an adapter: bind ``rank``/``alpha`` and apply the rank mask."""
        assert 0 < rank <= self.max_rank, (
            f"Adapter rank {rank} must be in (0, {self.max_rank}]"
        )
        self.alpha_values[idx] = alpha
        self.rank_values[idx] = rank
        self._apply_rank_mask(idx)

    def clear_adapter_slot(self, idx: int) -> None:
        """Free slot ``idx``: zero alpha, restore max rank, re-init weights."""
        self.alpha_values[idx] = 0
        self.rank_values[idx] = self.max_rank
        self.reset_adapter(idx)

    def _apply_rank_mask(self, idx: int) -> None:
        """Zero padded rows of A and padded cols of B for slot ``idx``.

        Plain ``nn.Linear``-backed adapters have no TP sharding, so the local
        cutoff equals the global ``actual_rank`` directly. With both sides
        zero in the padded region, the autograd chain through the two GEMMs
        keeps the gradient zero there too — no periodic re-masking needed.
        """
        actual_rank = int(self.rank_values[idx].item())
        if actual_rank >= self.max_rank:
            return
        adapter = self.adapters[idx]
        with torch.no_grad():
            adapter.linear_in.weight.data[actual_rank:].zero_()
            adapter.linear_out.weight.data[:, actual_rank:].zero_()

    def named_parameters_for_adapter(self, idx: int) -> Iterator[Tuple[str, nn.Parameter]]:
        prefix = f"adapters.{idx}."
        for name, param in self.adapters[idx].named_parameters():
            yield prefix + name, param

    def state_dict_for_adapter(self, idx: int, prefix: str = "") -> Dict[str, Any]:
        sd: Dict[str, Any] = {}
        self.adapters[idx].state_dict(destination=sd, prefix=f"{prefix}adapters.{idx}.")
        return sd

    def load_adapter(self, idx: int, state_dict: Dict[str, torch.Tensor]) -> None:
        self.adapters[idx].load_state_dict(state_dict, strict=True)


class MultiLoRALinear(AdapterWrapper):
    """Megatron parallel linear wrapped with *N* concurrent LoRA adapters.

    Each adapter slot is a :class:`ParallelLinearAdapter` stored in an
    ``nn.ModuleList``. Forward uses grouped GEMM with a single set of
    TP/SP comms for efficiency.

    For bridge export compatibility, use :func:`expose_adapter_slot` to
    temporarily expose one slot as ``.adapter``.
    """

    def __init__(
        self,
        to_wrap: nn.Module,
        n_adapters: int,
        dim: int,
        alpha: float,
        full_name: str,
        column_init_method: str = "xavier",
        row_init_method: str = "zero",
        dropout: float = 0.0,
        dropout_position: str = "pre",
        a2a_experimental: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.to_wrap = to_wrap
        self._adapter_enabled = True
        self.n_adapters = n_adapters
        self.max_rank = dim

        attrs = get_adapter_attributes_from_linear(to_wrap)

        # input_is_parallel distinguishes column-parallel base (False, e.g. linear_qkv,
        # linear_fc1) from row-parallel base (True, e.g. linear_proj, linear_fc2).
        # It controls which TP collective runs between the two grouped GEMMs and
        # whether the second GEMM's output needs to be all-gathered to match the
        # wrapped base linear's output layout.
        self.input_is_parallel = attrs.input_is_parallel
        self.disable_sequence_parallel_comm = attrs.disable_sequence_parallel_comm
        self.use_a2a = a2a_experimental
        self._gather_output = attrs.input_is_parallel

        # ModuleList of ParallelLinearAdapters gives per-adapter optimizer state
        # isolation, clean checkpoint serialization, and bridge export compatibility.
        self.adapters = nn.ModuleList([
            ParallelLinearAdapter(
                in_features=attrs.in_features,
                out_features=attrs.out_features,
                dim=dim,
                base_linear_name=full_name,
                activation="identity",
                alpha=alpha,
                input_is_parallel=attrs.input_is_parallel,
                column_init_method=column_init_method,
                row_init_method=row_init_method,
                disable_sequence_parallel_comm=attrs.disable_sequence_parallel_comm,
                a2a_experimental=a2a_experimental,
                dropout=dropout,
                dropout_position=dropout_position,
            )
            for _ in range(n_adapters)
        ])

        self.tokens_per_adapter: Optional[torch.Tensor] = None
        device = next(to_wrap.parameters()).device
        dtype = next(to_wrap.parameters()).dtype
        self.alpha_values = torch.ones(n_adapters, dtype=dtype, device=device)
        self.rank_values = torch.full((n_adapters,), dim, dtype=dtype, device=device)

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)

        if not self._adapter_enabled:
            return linear_output, bias

        tokens_per_adapter = self.tokens_per_adapter
        x = layernorm_output.contiguous()

        # SP gather (once) — for column-parallel base layers without an LN-fused
        # gather, the layernorm output is still SP-sharded and must be gathered
        # to full sequence before the adapter matmul.
        if not self.disable_sequence_parallel_comm and not self.input_is_parallel:
            x = gather_from_sequence_parallel_region(x)

        x_flat = x.reshape(-1, x.shape[-1])
        offsets = tokens_per_adapter.cumsum(dim=0, dtype=torch.int32)

        stacked_A = torch.stack([a.linear_in.weight for a in self.adapters])
        stacked_B = torch.stack([a.linear_out.weight for a in self.adapters])

        mid = torch._grouped_mm(x_flat, stacked_A.transpose(-2, -1), offsets)

        # TP collective between A and B: row-parallel base needs an all-reduce
        # of the partial sums; column-parallel base needs an all-gather of the
        # rank-sharded output to a full [tokens, dim] for the second GEMM.
        if self.input_is_parallel:
            mid = reduce_from_tensor_model_parallel_region(mid)
        else:
            mid = gather_from_tensor_model_parallel_region(mid)

        out = torch._grouped_mm(mid, stacked_B.transpose(-2, -1), offsets)

        # Per-token scaling is applied *before* the output-side TP/SP comms.
        # ``per_token_scaling`` is indexed by the full token count
        # (``tokens_per_adapter`` sums to it); doing it after a sequence-parallel
        # scatter would leave ``out`` with ``tokens/tp`` rows and crash here.
        scaling = self.alpha_values / self.rank_values
        per_token_scaling = torch.repeat_interleave(scaling, tokens_per_adapter).unsqueeze(-1)
        out = out * per_token_scaling

        # Match the wrapped base linear's output layout: row-parallel base
        # produces a fully-summed [tokens, h_out] tensor (which we then SP
        # scatter); column-parallel base keeps the [tokens, h_out/tp] shard.
        if self._gather_output:
            out = gather_from_tensor_model_parallel_region(out)

        if not self.disable_sequence_parallel_comm and self.input_is_parallel:
            if self.use_a2a:
                out = all2all_hp2sp(out)
            else:
                out = scatter_to_sequence_parallel_region(out)

        return linear_output + out.reshape(linear_output.shape), bias

    def reset_adapter(self, idx: int) -> None:
        from megatron.bridge.peft.utils import ParallelLinearAdapter
        col_fn = ParallelLinearAdapter._get_init_fn(None, "xavier")
        row_fn = ParallelLinearAdapter._get_init_fn(None, "zero")
        col_fn(self.adapters[idx].linear_in.weight.data)
        row_fn(self.adapters[idx].linear_out.weight.data)

    def init_adapter_slot(self, idx: int, rank: int, alpha: float) -> None:
        """Claim slot ``idx`` for an adapter: bind ``rank``/``alpha`` and apply the rank mask."""
        assert 0 < rank <= self.max_rank, (
            f"Adapter rank {rank} must be in (0, {self.max_rank}]"
        )
        self.alpha_values[idx] = alpha
        self.rank_values[idx] = rank
        self._apply_rank_mask(idx)

    def clear_adapter_slot(self, idx: int) -> None:
        """Free slot ``idx``: zero alpha, restore max rank, re-init weights."""
        self.alpha_values[idx] = 0
        self.rank_values[idx] = self.max_rank
        self.reset_adapter(idx)

    def _apply_rank_mask(self, idx: int) -> None:
        """Zero padded rows of A and padded cols of B for slot ``idx``.

        For column-parallel base layers (``linear_qkv``, ``linear_fc1``)
        ``linear_in.weight`` is sharded across TP — rank ``r`` owns global
        rows ``[r*L : (r+1)*L]`` where ``L = max_rank/tp``. For row-parallel
        base it is replicated. Map the global cutoff ``actual_rank`` into
        the local shard before zeroing.

        With both sides zero in the padded region, the autograd chain through
        the two GEMMs keeps the gradient zero there too — no periodic
        re-masking needed during training.
        """
        actual_rank = int(self.rank_values[idx].item())
        if actual_rank >= self.max_rank:
            return
        adapter = self.adapters[idx]
        local_rank_dim = adapter.linear_in.weight.shape[0]
        if local_rank_dim < self.max_rank:
            tp_rank = parallel_state.get_tensor_model_parallel_rank()
            shard_start = tp_rank * local_rank_dim
            local_start = max(0, actual_rank - shard_start)
        else:
            local_start = actual_rank
        with torch.no_grad():
            if local_start < local_rank_dim:
                adapter.linear_in.weight.data[local_start:].zero_()
            adapter.linear_out.weight.data[:, actual_rank:].zero_()

    def state_dict(
        self,
        destination: Optional[Dict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        if destination is None:
            destination = {}
        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.adapters.state_dict(destination=destination, prefix=f"{prefix}adapters.", keep_vars=keep_vars)
        return destination

    def sharded_state_dict(
        self,
        prefix: str = "",
        sharded_offsets: Tuple[Tuple[int, int, int], ...] = (),
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sharded_sd: Dict[str, Any] = {}
        sharded_sd.update(self.to_wrap.sharded_state_dict(prefix, sharded_offsets, metadata))
        for k, v in self.adapters.state_dict(prefix=f"{prefix}adapters.").items():
            sharded_sd[k] = v
        return sharded_sd


# ==================================================================
# Standalone functions
# ==================================================================

_MULTI_LORA_TYPES = (MultiLoRALinear, SimpleMultiLoRALinear)


def _iter_multi_lora_modules(model):
    models = model if isinstance(model, list) else [model]
    for model_chunk in models:
        for module in model_chunk.modules():
            if isinstance(module, _MULTI_LORA_TYPES):
                yield module


def set_tokens_per_adapter_slot(model, tokens_per_adapter: torch.Tensor) -> None:
    """Route a packed micro-batch to its per-slot token spans.

    ``tokens_per_adapter[i]`` is the number of contiguous tokens in the
    upcoming forward that belong to adapter slot ``i``. Must sum to the total
    token count of the micro-batch.
    """
    for module in _iter_multi_lora_modules(model):
        module.tokens_per_adapter = tokens_per_adapter


def init_adapter_slot(model, idx: int, rank: int, alpha: float) -> None:
    """Claim slot ``idx`` across every multi-LoRA layer for an adapter.

    A model-wide adapter is the set of slot-``idx`` chunks across all layers;
    this initialises that set with the given ``rank``/``alpha``. Thin iterator
    over the model — per-slot setup (rank/alpha bookkeeping + rank-mask
    invariant) lives on the layer itself in
    :meth:`MultiLoRALinear.init_adapter_slot` /
    :meth:`SimpleMultiLoRALinear.init_adapter_slot`.
    """
    for module in _iter_multi_lora_modules(model):
        module.init_adapter_slot(idx, rank, alpha)


def clear_adapter_slot(model, idx: int) -> None:
    """Release slot ``idx`` across every multi-LoRA layer (zero alpha, re-init weights)."""
    for module in _iter_multi_lora_modules(model):
        module.clear_adapter_slot(idx)


def load_adapter(model, idx: int, state_dict: Dict[str, torch.Tensor]) -> int:
    """Load Megatron-shard format adapter weights into slot ``idx``.

    ``state_dict`` must use the *Megatron-native* names produced by saving
    while ``expose_adapter_slot(model, idx)`` is active — i.e. the same
    layout this function constructs to look them up. Each tensor is the
    local TP/PP shard, copied straight into the slot parameter with no
    gather, scatter, or rank-padding logic.

    Saving from slot A and loading into slot B is fine because the slot
    index is stripped from the name (``...adapter.linear_in.weight``)
    while ``expose_adapter_slot`` is active.

    Returns the number of tensors loaded (for logging / sanity checks).
    """
    loaded = 0
    with expose_adapter_slot(model, idx):
        models = model if isinstance(model, list) else [model]
        for chunk in models:
            for name, param in chunk.named_parameters():
                if ".adapter." not in name:
                    continue
                if name not in state_dict:
                    continue
                src = state_dict[name].to(device=param.device, dtype=param.dtype)
                param.data.copy_(src)
                loaded += 1
    return loaded


def expose_adapter_slot(model, idx: int):
    """Context manager that temporarily exposes one adapter slot as ``.adapter``.

    Used by two consumers:

    * The bridge's ``export_adapter_weights`` looks for ``.adapter.linear_in.weight``
      (single-LoRA layout) on each wrapped module.
    * Megatron-native save/load walk ``model.named_parameters()`` and want names
      that don't contain the slot index, so saving from slot ``A`` and loading into
      slot ``B`` produces matching keys.

    Both ``MultiLoRALinear`` and ``SimpleMultiLoRALinear`` are handled via the
    common ``.adapters`` ModuleList — duck-typed rather than ``isinstance``-checked
    so future multi-LoRA module types are picked up automatically.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        modules = list(_iter_multi_lora_modules(model))
        saved = {}
        for m in modules:
            if "adapters" in m._modules:
                saved[id(m)] = m._modules.pop("adapters")
                m.adapter = saved[id(m)][idx]

        yield

        for m in modules:
            if id(m) in saved:
                if "adapter" in m._modules:
                    del m._modules["adapter"]
                m._modules["adapters"] = saved[id(m)]

    return _ctx()


def hide_adapters(model):
    """Context manager that temporarily hides all adapter params from the model.

    Used during base checkpoint loading so the bridge doesn't try to map
    adapter parameters to HF weights.
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        modules = list(_iter_multi_lora_modules(model))
        saved = {}
        for m in modules:
            if isinstance(m, MultiLoRALinear) and "adapters" in m._modules:
                saved[id(m)] = m._modules.pop("adapters")
            elif isinstance(m, SimpleMultiLoRALinear) and "adapters" in m._modules:
                saved[id(m)] = m._modules.pop("adapters")
        yield
        for m in modules:
            if id(m) in saved:
                m._modules["adapters"] = saved[id(m)]

    return _ctx()
