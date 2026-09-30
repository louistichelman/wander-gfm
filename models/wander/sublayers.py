from typing import Callable, Optional, Tuple
import os

import torch
from torch import nn, Tensor
import torch.distributed as dist
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover
    SDPBackend = None
    sdpa_kernel = None

_LOG_WANDER_MEM = os.environ.get("LOG_WANDER_MEM", "0") == "1"


def materialize_if_broadcast(x: Tensor) -> Tensor:
    """Copy expand/as_strided views into unique storage.

    NC initializes every node from one role-embedding row via ``expand()``.
    Intra-node attention materializes unique rows as a side effect; with intra
    off the 0-stride view can reach global SDPA and force the math kernel
    (full ``[B,H,N,N]`` attention matrix) on backward.
    """
    if not isinstance(x, Tensor) or x.numel() == 0:
        return x
    try:
        storage_numel = x.untyped_storage().size() // max(x.element_size(), 1)
    except Exception:
        return x.contiguous()
    if storage_numel < x.numel():
        return x.contiguous()
    return x

# torch.compile of SwiGLU / GRU residual-FFN. One compiled function per
# kernel, shared across BidirectionalGRU copies (T × channels).
# Standalone RMSNorm stays eager: CoDEx intra is S=1 with ~760 tiny
# launches, and compiling it made intra 55 ms -> 1.7–2.4 s. GRU still
# inlines RMSNorm inside the compiled mix-FFN.
# Not CUDA-graphing the 6 refine steps: walk buffers, prune, and K-chunks
# change addresses/shapes every forward.
_KERNEL_COMPILE = False
_COMPILED: dict[str, Callable] = {}
_COMPILE_LOGGED = False


def set_kernel_compile(enabled: bool) -> None:
    global _KERNEL_COMPILE, _COMPILE_LOGGED
    enabled = bool(enabled)
    if enabled and not _KERNEL_COMPILE and not _COMPILE_LOGGED:
        if not dist.is_initialized() or dist.get_rank() == 0:
            print("[wander] eval torch.compile enabled for SwiGLU / GRU residual-FFN")
        _COMPILE_LOGGED = True
    _KERNEL_COMPILE = enabled


def kernel_compile_enabled() -> bool:
    return _KERNEL_COMPILE


def _call_compiled(name: str, eager_fn: Callable, *args):
    if not _KERNEL_COMPILE:
        return eager_fn(*args)
    fn = _COMPILED.get(name)
    if fn is None:
        try:
            fn = torch.compile(eager_fn, dynamic=True)
        except Exception as exc:
            print(f"[wander] torch.compile({name}) failed ({exc}); using eager")
            fn = eager_fn
        _COMPILED[name] = fn
    try:
        return fn(*args)
    except Exception as exc:
        print(f"[wander] compiled {name} raised ({exc}); falling back to eager")
        _COMPILED[name] = eager_fn
        return eager_fn(*args)


def _rmsnorm_eager(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return output.type_as(x) * weight


def _swiglu_eager(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    return F.linear(F.silu(F.linear(x, w1)) * F.linear(x, w3), w2)


def _gru_mix_ffn_eager(
    residual: Tensor,
    gru_h: Tensor,
    gru_out_weight: Tensor,
    ffn_norm_weight: Tensor,
    ffn_w1: Tensor,
    ffn_w2: Tensor,
    ffn_w3: Tensor,
    eps: float,
) -> Tensor:
    x = residual + F.linear(gru_h, gru_out_weight)
    xn = _rmsnorm_eager(x, ffn_norm_weight, eps)
    return x + _swiglu_eager(xn, ffn_w1, ffn_w2, ffn_w3)


class RMSNorm(nn.Module):
    """https://github.com/meta-llama/llama3/blob/main/llama/model.py"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    """https://github.com/meta-llama/llama3/blob/main/llama/model.py"""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        if _KERNEL_COMPILE:
            return _call_compiled(
                "swiglu", _swiglu_eager, x, self.w1.weight, self.w2.weight, self.w3.weight
            )
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: Tensor,
    xk: Tensor,
    freqs_cis: Tensor,
    rope_positions: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    # xq, xk: [B, n_heads, S, head_dim]  freqs_cis: [max_pos, head_dim//2]
    if rope_positions is not None:
        freqs_cis = freqs_cis[rope_positions]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[None, None, :, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class BidirectionalGRU(nn.Module):
    def __init__(self, dim: int, n_layers: int, multiple_of: int, norm_eps: float):
        super().__init__()
        self.gru_norm = RMSNorm(dim, eps=norm_eps)
        self.gru = nn.GRU(
            dim, dim, num_layers=n_layers, batch_first=True, bidirectional=True
        )
        self.gru_out = nn.Linear(2 * dim, dim, bias=False)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        gru_h = self.gru(self.gru_norm(x))[0]
        if _KERNEL_COMPILE:
            return _call_compiled(
                "gru_mix_ffn",
                _gru_mix_ffn_eager,
                x,
                gru_h,
                self.gru_out.weight,
                self.ffn_norm.weight,
                self.feed_forward.w1.weight,
                self.feed_forward.w2.weight,
                self.feed_forward.w3.weight,
                self.ffn_norm.eps,
            )
        x = x + self.gru_out(gru_h)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class SequenceAttentionLayer(nn.Module):
    def __init__(self, dim: int, n_heads: int, multiple_of: int, norm_eps: float, use_rotary_embedding: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rotary_embedding = use_rotary_embedding

        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=None,
        )

        self._capture_attention: bool = False
        self._attention_weights: list = []

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_gather_idx: Optional[Tensor] = None,
        kv_key_padding_mask: Optional[Tensor] = None,
        external_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        """
        Args:
            x: [B, S, D]
            attn_mask: optional boolean [B, S]. True = attend, False = mask out.
                       Expanded to [B, 1, 1, S] for key masking in SDPA.
            kv_gather_idx: optional long [B, M] selecting KV source positions
                       per row (padded to a common M). When provided, K/V are
                       projected only from the gathered slice of ``x_normed``,
                       so ``wk`` / ``wv`` skip the masked positions entirely.
                       Incompatible with ``use_rotary_embedding=True``.
            kv_key_padding_mask: optional boolean [B, M] paired with
                       ``kv_gather_idx``. True = real key, False = padding.
                       When ``kv_gather_idx`` is given, ``attn_mask`` is
                       ignored (this mask takes its place along the M axis).
            external_kv: optional pre-projected ``(k, v)``, each [B, M, D]
                       (i.e. outputs of ``wk`` / ``wv`` on already-normed
                       states). When provided, K/V projection is skipped
                       entirely; ``attn_mask`` and ``kv_gather_idx`` must be
                       None. Incompatible with ``use_rotary_embedding=True``.
        """
        x = x.contiguous()
        B, S, D = x.shape
        residual = x
        x_normed = self.attn_norm(x)
        q_src = x_normed
        q_residual = residual
        q_len = S
        q = self.wq(q_src).view(B, q_len, self.n_heads, self.head_dim).transpose(1, 2)

        if external_kv is not None:
            assert not self.use_rotary_embedding, (
                "external_kv is incompatible with use_rotary_embedding=True"
            )
            assert kv_gather_idx is None and attn_mask is None, (
                "external_kv is mutually exclusive with kv_gather_idx / attn_mask"
            )
            k_ext, v_ext = external_kv
            k_ext = k_ext.to(dtype=q.dtype)
            v_ext = v_ext.to(dtype=q.dtype)
            M = k_ext.shape[1]
            k = k_ext.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
            v = v_ext.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
            mask_4d = (
                kv_key_padding_mask[:, None, None, :]
                if kv_key_padding_mask is not None
                else None
            )
        else:
            if kv_gather_idx is None:
                k_src = x_normed
                mask_4d = attn_mask[:, None, None, :] if attn_mask is not None else None
            else:
                assert not self.use_rotary_embedding, (
                    "kv_gather_idx is incompatible with use_rotary_embedding=True"
                )
                idx = kv_gather_idx[:, :, None].expand(-1, -1, D)
                k_src = x_normed.gather(1, idx)  # [B, M, D]
                mask_4d = (
                    kv_key_padding_mask[:, None, None, :]
                    if kv_key_padding_mask is not None
                    else None
                )
            M = k_src.shape[1]
            k = self.wk(k_src).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
            v = self.wv(k_src).view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_rotary_embedding:
            freqs_cis = precompute_freqs_cis(self.head_dim, S, device=x.device)
            q, k = apply_rotary_emb(q, k, freqs_cis)

        # An all-True boolean mask still forces the math SDPA kernel (full
        # [B,H,Q,M] matrix). bsize=1 NC never has KV padding, so drop it.
        if mask_4d is not None and mask_4d.dtype == torch.bool and bool(mask_4d.all().item()):
            mask_4d = None

        if _LOG_WANDER_MEM and (B <= 4 or q_len >= 4096):
            alloc = (
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
            )
            print(
                f"[wander_sdpa] B={B} heads={self.n_heads} q_len={q_len} M={int(k.shape[-2])} "
                f"x.stride={tuple(x.stride())} x.contiguous={x.is_contiguous()} "
                f"has_attn_mask={mask_4d is not None} dtype={q.dtype} alloc_gb={alloc:.2f}",
                flush=True,
            )
        if self._capture_attention:
            scale = self.head_dim ** -0.5
            aw = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask_4d is not None:
                aw = aw.masked_fill(~mask_4d, float('-inf'))
            aw = F.softmax(aw, dim=-1)
            self._attention_weights.append(aw.detach().cpu().to(torch.float32))
            attn = torch.matmul(aw, v)
        elif sdpa_kernel is not None:
            with sdpa_kernel(
                [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
            ):
                attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d)
        attn = attn.transpose(1, 2).contiguous().view(B, q_len, D)
        y = q_residual + self.wo(attn)
        y = y + self.feed_forward(self.ffn_norm(y))
        return y


class SequenceAttention(nn.Module):
    def __init__(self, dim: int, n_layers: int, n_heads: int, multiple_of: int, norm_eps: float, use_rotary_embedding: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            SequenceAttentionLayer(dim, n_heads, multiple_of, norm_eps, use_rotary_embedding)
            for _ in range(n_layers)
        ])

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        kv_gather_idx: Optional[Tensor] = None,
        kv_key_padding_mask: Optional[Tensor] = None,
        external_kv: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tensor:
        if external_kv is not None:
            assert len(self.layers) == 1, (
                "external_kv requires a single attention layer (n_layers == 1); "
                "with stacked layers the KV states evolve between sublayers."
            )
        for layer in self.layers:
            x = layer(
                x,
                attn_mask=attn_mask,
                kv_gather_idx=kv_gather_idx,
                kv_key_padding_mask=kv_key_padding_mask,
                external_kv=external_kv,
            )
        return x


class IntraNodeAttention(nn.Module):
    """Self-attention that mixes feature, label, and node embeddings
    within each node.

    Tokens per node: ``[node_emb, feat_0, ..., feat_{F-1}, label_emb]``

    Callers are expected to chunk over the ``B*N`` node dimension themselves
    and call :meth:`forward_tokens` with ``[C, S, D]`` chunks.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        multiple_of: int,
        norm_eps: float = 1e-5,
        ffn_dim_multiplier: Optional[float] = None,
        use_rotary_embedding: bool = False,
        use_ffn: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.use_rotary_embedding = use_rotary_embedding
        self.use_ffn = use_ffn

        self.attn_norm = RMSNorm(dim, eps=norm_eps)
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(
            dim=dim,
            hidden_dim=4 * dim,
            multiple_of=multiple_of,
            ffn_dim_multiplier=ffn_dim_multiplier,
        )

        self._capture_attention: bool = False
        self._attention_weights: list = []

    def forward_tokens(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Attention (+ optional FFN) on a token block of shape ``[C, S, D]``.

        Args:
            attn_mask: optional boolean ``[S, S]``. True = attend, False = mask out.
        """
        S = x.shape[1]
        residual = x
        x_normed = self.attn_norm(x)
        q = self.wq(x_normed).view(-1, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x_normed).view(-1, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x_normed).view(-1, S, self.n_heads, self.head_dim).transpose(1, 2)
        if self.use_rotary_embedding and S > 1:
            freqs_cis = precompute_freqs_cis(self.head_dim, S, device=x.device)
            q, k = apply_rotary_emb(q, k, freqs_cis)
        mask_4d = attn_mask[None, None, :, :] if attn_mask is not None else None
        if S == 1:
            attn = v
        elif self._capture_attention:
            scale = self.head_dim ** -0.5
            aw = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask_4d is not None:
                aw = aw.masked_fill(~mask_4d, float("-inf"))
            aw = F.softmax(aw, dim=-1)
            self._attention_weights.append(aw.detach().cpu().to(torch.float32))
            attn = torch.matmul(aw, v)
        else:
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d)
        attn = attn.transpose(1, 2).contiguous().view(-1, S, self.dim)
        x = residual + self.wo(attn)
        if self.use_ffn:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x
