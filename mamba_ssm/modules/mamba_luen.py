# Copyright (c) 2023, Tri Dao, Albert Gu.
# Modified to include a diagonal-approx parallel Luenberger Observer within Mamba SSM
# Keeps selective_scan parallelism by using state augmentation and readout blending.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None


# =========================
# Luenberger Observer (diag)
# =========================

class LuenbergerObserver(nn.Module):
    """
    Diagonal-approx parallel Luenberger observer for SSM state estimation.

    We create two SSM branches and blend their readouts:
      - Original:  x[k+1]  = A * x[k] + B * u[k]
      - Observer:  xh[k+1] = (A - gamma) * xh[k] + (B + gamma * D_mean) * u[k]
    Final readout:
      y[k] = <[(1 - alpha) * C, alpha * C], [x[k], xh[k]]> + D * u[k]

    This design preserves per-channel, per-state independence required by selective_scan.
    """

    def __init__(self, d_inner: int, d_state: int, dt_rank: int, device=None, dtype=None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        # Learnable diagonal observer gain per state (broadcast along channels)
        self.observer_gain = nn.Parameter(
            torch.randn(d_state, device=device, dtype=dtype) * 0.01
        )

    @torch.no_grad()
    def reset_state(self, *args, **kwargs):
        # Kept for API compatibility; state is managed by the SSM kernels.
        return

    def create_augmented_matrices(
        self,
        A: Tensor,        # (d_inner, d_state)
        B: Tensor,        # (batch, d_state, L)
        C: Tensor,        # (batch, d_state, L)
        D: Tensor,        # (d_inner,)
        alpha: float,     # blending weight in [0, 1]
    ):
        """
        Returns augmented diagonal SSM parameters that can be used by selective_scan_fn:
            A_aug: (d_inner, 2*d_state)
            B_aug: (batch, 2*d_state, L)
            C_aug: (batch, 2*d_state, L)
        """
        assert A.dim() == 2 and B.dim() == 3 and C.dim() == 3 and D.dim() == 1
        batch, n, L = B.shape
        d_inner, nA = A.shape
        assert nA == n, "Mismatch between A and B/C d_state"

        device, dtype = A.device, A.dtype
        n_aug = 2 * n

        # Non-negative per-state gain; broadcast across channels and time
        gamma = F.softplus(self.observer_gain)              # (n,)
        gamma_row = gamma.view(1, n).expand(d_inner, -1)    # (d_inner, n)

        # A_aug
        A_aug = torch.zeros(d_inner, n_aug, device=device, dtype=dtype)
        A_aug[:, :n] = A
        A_aug[:, n:] = A - gamma_row                         # observer branch

        # B_aug
        B_aug = torch.zeros(batch, n_aug, L, device=device, dtype=dtype)
        B_aug[:, :n, :] = B
        # Use scalar approximation for D to keep shapes aligned with (batch, n, L)
        d_scalar = D.mean()                                  # ()
        gamma_D = (gamma * d_scalar).view(1, n, 1)           # (1, n, 1)
        B_aug[:, n:, :] = B + gamma_D                        # observer branch: B + gamma * D_mean

        # C_aug: linear blending at readout with provided alpha
        C_aug = torch.zeros(batch, n_aug, L, device=device, dtype=dtype)
        C_aug[:, :n, :] = (1.0 - alpha) * C
        C_aug[:, n:, :] = alpha * C

        return A_aug, B_aug, C_aug


# =======================================
# Mamba with Luenberger Observer (diag)
# =======================================

class MambaWithLuenbergerObserver(nn.Module):
    """
    Mamba layer with a diagonal-approx parallel Luenberger observer implemented via state augmentation.
    When use_observer=True, the internal state dimension is doubled (2 * d_state) and
    selective_scan_fn runs once on the augmented SSM.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        use_observer: bool = True,
        observer_alpha: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.layer_idx = layer_idx
        self.use_observer = bool(use_observer)
        self.observer_alpha = float(observer_alpha)

        # When observer is on, we double the latent state size
        self.d_state_actual = (2 * d_state) if self.use_observer else d_state

        # Fast path only safe when observer is disabled
        self.use_fast_path = bool(use_fast_path) and (not self.use_observer)

        # Projections
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state_actual * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # dt initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError("dt_init must be 'constant' or 'random'")

        # Set dt bias so that softplus(dt_bias) in [dt_min, dt_max]
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        # S4D real init (per-channel diagonal A in log-space)
        A = repeat(
            torch.arange(1, self.d_state_actual + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # keep in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D skip
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        # Observer module
        if self.use_observer:
            self.observer = LuenbergerObserver(
                d_inner=self.d_inner,
                d_state=self.d_state,
                dt_rank=self.dt_rank,
                device=device,
                dtype=dtype,
            )

    def forward(self, hidden_states: Tensor, inference_params=None):
        """
        hidden_states: (B, L, D_model)
        returns: (B, L, D_model)
        """
        bsz, seqlen, _ = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, bsz)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # Project and reshape to HBL
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        # Diagonal A
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state_actual)

        # Fast path when observer is disabled
        if self.use_fast_path and not self.use_observer:
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,
                None,
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            return out

        # Convolution + activation
        x, z = xz.chunk(2, dim=1)
        if conv_state is not None:
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))

        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        # Input-dependent parameters
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl, d_inner_actual)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state_actual, self.d_state_actual], dim=-1)

        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)

        B = rearrange(B, "(b l) n -> b n l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=seqlen).contiguous()

        # Build augmented SSM if observer is enabled
        if self.use_observer:
            # Use the first half (original) to construct augmented params
            B_orig = B[:, :self.d_state, :]
            C_orig = C[:, :self.d_state, :]

            A_aug, B_aug, C_aug = self.observer.create_augmented_matrices(
                A[:, :self.d_state],   # (d_inner, d_state)
                B_orig,                # (B, d_state, L)
                C_orig,                # (B, d_state, L)
                self.D,                # (d_inner,)
                alpha=self.observer_alpha,
            )
            A_final, B_final, C_final = A_aug, B_aug, C_aug
        else:
            A_final, B_final, C_final = A, B, C

        # Run selective scan
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A_final,
            B_final,
            C_final,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)

        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        """
        Single-token decoding step. For simplicity, observer augmentation is not applied here.
        """
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only 1-token decoding supported"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B, 2*D_inner)
        x, z = xz.chunk(2, dim=-1)

        # Conv state update
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x, conv_state, rearrange(self.conv1d.weight, "d 1 w -> d w"), self.conv1d.bias, self.activation
            )

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state_actual, self.d_state_actual], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        # Observer augmentation is omitted here for simplicity
        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state_actual, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states: bool = False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state_actual,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    @torch.no_grad()
    def reset_observer(self):
        if self.use_observer:
            self.observer.reset_state()


class MambaBlockWithObserver(nn.Module):
    """
    Multi-layer Mamba with Luenberger Observers (diagonal-approx), pre-norm + residual.
    """
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 768,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_observer: bool = True,
        observer_alpha: float = 0.1,
        **mamba_kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_observer = use_observer

        self.mamba_layers = nn.ModuleList([
            MambaWithLuenbergerObserver(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=i,
                use_observer=use_observer,
                observer_alpha=observer_alpha,
                **mamba_kwargs,
            )
            for i in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])

    def forward(self, x: Tensor, inference_params=None):
        """
        x: (B, L, D_model)
        """
        for mamba_layer, layer_norm in zip(self.mamba_layers, self.layer_norms):
            residual = x
            x = layer_norm(x)
            x = mamba_layer(x, inference_params)
            x = x + residual
        return x

    @torch.no_grad()
    def reset_observers(self):
        if self.use_observer:
            for m in self.mamba_layers:
                m.reset_observer()
