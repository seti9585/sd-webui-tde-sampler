"""
TDE Sampler — reForge Extension
================================
Location: extensions/tde-sampler/scripts/tde_sampler.py

Port of the torchdiffeq-based ODE sampler from
ComfyUI-ODE (https://github.com/redhottensors/ComfyUI-ODE) to reForge.

[Differences from reForge's built-in ODE Custom]
- Although reForge's ODE Custom also uses torchdiffeq, this extension
  registers as an independent sampler, allowing different methods to be
  selected for txt2img and hires.fix.
- Method, rtol, and atol can be changed on the fly via the Script UI.

[Note on Forge Neo compatibility]
  reForge:  from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
  Forge Neo: from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
  (Both paths are now the same; this note is kept for reference.)

Dependencies:
  torchdiffeq  (pip install torchdiffeq)
"""

from __future__ import annotations

import logging
import os
import sys

import gradio as gr
import torch
from tqdm.auto import tqdm, trange

# ---------------------------------------------------------------------------
# Add extension directory to sys.path (same as rk_core)
# ---------------------------------------------------------------------------
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

# ---------------------------------------------------------------------------
# reForge core modules
# ---------------------------------------------------------------------------
from modules import sd_samplers_common, shared, script_callbacks
from modules.shared import opts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torchdiffeq availability check
# ---------------------------------------------------------------------------
try:
    import torchdiffeq
    HAS_TORCHDIFFEQ = True
except ModuleNotFoundError:
    HAS_TORCHDIFFEQ = False
    logger.error(
        "[TDE Sampler] torchdiffeq not found. Please run: pip install torchdiffeq"
    )

# ---------------------------------------------------------------------------
# MPS availability check
# ---------------------------------------------------------------------------
HAS_MPS = torch.backends.mps.is_available()


# ===========================================================================
# Section 0: Method Definitions
# ===========================================================================

ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS    = {"euler", "midpoint", "rk4", "heun3"}
ALL_SOLVERS      = sorted([*ADAPTIVE_SOLVERS, *FIXED_SOLVERS])

# Options keys
OPT_LOG_RTOL  = "tde_sampler_log_rtol"
OPT_LOG_ATOL  = "tde_sampler_log_atol"
OPT_MAX_STEPS = "tde_sampler_max_steps"

# Default values (matching reForge's ODE Custom)
DEF_LOG_RTOL  = -3.0   # matches reForge ODE Custom default
DEF_LOG_ATOL  = -4.0   # matches reForge ODE Custom default
DEF_MAX_STEPS = 250

# Slider references for forced update on page load
_tde_max_steps_sliders = []


def _resolve_tde_log_rtol(p):
    if getattr(p, "is_hr_pass", False):
        return getattr(
            p, "_tde_hr_log_rtol",
            getattr(p, "_tde_txt2img_log_rtol", _get(OPT_LOG_RTOL, DEF_LOG_RTOL))
        )
    return getattr(p, "_tde_txt2img_log_rtol", _get(OPT_LOG_RTOL, DEF_LOG_RTOL))


def _resolve_tde_log_atol(p):
    if getattr(p, "is_hr_pass", False):
        return getattr(
            p, "_tde_hr_log_atol",
            getattr(p, "_tde_txt2img_log_atol", _get(OPT_LOG_ATOL, DEF_LOG_ATOL))
        )
    return getattr(p, "_tde_txt2img_log_atol", _get(OPT_LOG_ATOL, DEF_LOG_ATOL))


def _get(key, default):
    return getattr(opts, key, default)


# ---------------------------------------------------------------------------
# Flow Matching detection and noise injection helpers  [patched]
# ---------------------------------------------------------------------------

def _is_flow_matching(model_wrap):
    # Check whether the loaded model uses Flow Matching
    # (Anima/DiT, FLUX, SD3, ...) or standard DDPM/EDM (SDXL, SD1.5, ...).
    #
    # Strategy 1: inspect the model_sampling class name.
    # Strategy 2: fallback - Flow Matching models have sigma_max <= 1.0,
    #             while DDPM/EDM models have sigma_max ~ 14.6.
    #             Using 1.5 as a safe threshold.
    #
    # NOTE: detection uses model_wrap.sigmas[-1] (the model's inherent sigma
    # table maximum), NOT sigma_sched[0] (which varies with denoising_strength).
    # This guarantees correct behaviour at all denoising_strength values.
    inner = getattr(model_wrap, 'inner_model', model_wrap)
    ms = getattr(inner, 'model_sampling', None)
    if ms is not None:
        ms_type = type(ms).__name__
        if any(kw in ms_type for kw in ('Flow', 'Flux')):
            return True
    sigs = getattr(model_wrap, 'sigmas', None)
    if sigs is not None and len(sigs) > 0:
        return sigs[-1].item() <= 1.5
    return False


def _flow_aware_noise_injection(x, noise, sigma, model_wrap):
    # Apply the correct noise injection for the model type.
    #
    # Flow Matching: x_t = (1 - t) * x + t * noise   (linear interpolation)
    # DDPM / EDM:    x_t = x + sigma * noise           (additive)
    if _is_flow_matching(model_wrap):
        t = sigma.clamp(0.0, 1.0)
        return (1.0 - t) * x + t * noise
    return x + noise * sigma



# ===========================================================================
# Section 1: ODE Right-Hand Side Function (ported from ComfyUI-ODE's ODEFunction)
# ===========================================================================

class TDEODEFunction:
    """
    Right-hand side function of the ODE, called by torchdiffeq.odeint.
    Processes one sample at a time (torchdiffeq specification).

    Probability flow ODE:
        dx/dσ = (x - D(x, σ)) / σ
    """

    def __init__(
        self,
        model,
        t_min: float,
        t_max: float,
        n_steps: int,
        is_adaptive: bool,
        extra_args: dict | None = None,
        callback=None,
        cfg_denoiser=None,
        pbar=None,
    ):
        self.model        = model
        self.t_min        = t_min
        self.t_max        = t_max
        self.n_steps      = n_steps
        self.is_adaptive  = is_adaptive
        self.extra_args   = extra_args or {}
        self.callback     = callback
        self.cfg_denoiser = cfg_denoiser
        self.pbar         = pbar
        self.step         = 0

    def __call__(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t : scalar tensor  current sigma value
            y : (C*H*W,) tensor  current latent (flattened, single sample)
        Returns:
            dy/dt : (C*H*W,) tensor
        """
        if t <= 1e-5:
            return torch.zeros_like(y)

        # torchdiffeq processes one sample at a time; unsqueeze to add batch dimension.
        #
        # The ODE solver may integrate in float64 (adaptive methods on CUDA).
        # All denoising model internals and pre-/post-CFG hooks (SkimmedCFG,
        # Mahiro CFG, AutomaticCFG, etc.) expect float32 tensors.  Passing
        # float64 latents causes dtype mismatches in hook computations —
        # sign-based mask comparisons (SkimmedCFG) produce incorrect masks at
        # high CFG values, leading to corrupted images.
        #
        # Cast to float32 for the model call; convert the result back to the
        # ODE dtype so the integration remains numerically consistent.
        if y.dtype != torch.float32:
            y_fp32 = y.to(dtype=torch.float32)
            t_fp32 = t.to(dtype=torch.float32)
            denoised = self.model(
                y_fp32.unsqueeze(0),
                t_fp32.unsqueeze(0),
                **self.extra_args
            )
            return (y - denoised.squeeze(0).to(dtype=y.dtype)) / t

        denoised = self.model(
            y.unsqueeze(0),
            t.unsqueeze(0),
            **self.extra_args
        )
        return (y - denoised.squeeze(0)) / t

    def callback_step(self, t0, y0, dt):
        """Fixed step: called after each step."""
        if self.is_adaptive:
            return

        self._fire_callback(t0, y0, self.step)

        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"σ": f"{t0.item():.4f}"})
            self.pbar.refresh()

        # Update cfg_denoiser.step
        if self.cfg_denoiser is not None and hasattr(self.cfg_denoiser, "step"):
            total = getattr(self.cfg_denoiser, "total_steps", None)
            if total is not None:
                self.cfg_denoiser.step = min(self.step, total - 1)
            else:
                self.cfg_denoiser.step = self.step

        self.step += 1

    def callback_accept_step(self, t0, y0, dt):
        """Adaptive step: called after each accepted step."""
        if not self.is_adaptive:
            return

        progress = (self.t_max - t0.item()) / max(self.t_max - self.t_min, 1e-8)
        i = round((self.n_steps - 1) * progress)

        self._fire_callback(t0, y0, i)

        if self.pbar is not None:
            new_step = round(100 * progress)
            self.pbar.update(new_step - self.step)
            self.step = new_step
            self.pbar.set_postfix({"σ": f"{t0.item():.4f}"})
            self.pbar.refresh()

        # Update cfg_denoiser.step
        if self.cfg_denoiser is not None and hasattr(self.cfg_denoiser, "step"):
            total = getattr(self.cfg_denoiser, "total_steps", None)
            if total is not None:
                self.cfg_denoiser.step = min(i, total - 1)
            else:
                self.cfg_denoiser.step = i

    def _fire_callback(self, t0, y0, i):
        if self.callback is None:
            return
        self.callback({
            "x":         y0.unsqueeze(0),
            "i":         i,
            "sigma":     t0,
            "sigma_hat": t0,
            "denoised":  y0.unsqueeze(0),
        })

    def reset(self):
        self.step = 0
        if self.pbar is not None:
            self.pbar.reset()


# ===========================================================================
# Section 2: Core Sampling Function
# ===========================================================================

def _run_tde_sampler(
    solver: str,
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args: dict | None = None,
    callback=None,
    cfg_denoiser=None,
    log_rtol: float = DEF_LOG_RTOL,
    log_atol: float = DEF_LOG_ATOL,
    max_steps: int  = DEF_MAX_STEPS,
):
    """
    Solve the ODE using torchdiffeq.odeint.
    Equivalent to ComfyUI-ODE's ODESampler.__call__().
    Processes one sample at a time.
    """
    is_adaptive = solver in ADAPTIVE_SOLVERS
    t_max   = sigmas.max()
    t_min   = sigmas.min()
    n_steps = len(sigmas)
    batch   = x.shape[0]

    # Dtype for ODE integration.
    # Adaptive solvers (dopri5, bosh3, etc.) use float64 on CUDA for accurate
    # per-step error estimation.  Fixed-step solvers (euler, rk4, heun3, etc.)
    # use float32 to match standard k-diffusion behaviour: float64 integration
    # accumulates rounding differences over the sigma schedule and produces
    # sparkle / grain artifacts, most visible in hires.fix passes where the
    # sigma range is small.
    if HAS_MPS:
        ode_dtype = torch.float32
    elif is_adaptive:
        ode_dtype = torch.float64
    else:
        ode_dtype = torch.float32

    # sigma schedule (fixed steps) or [t_max, t_min] (adaptive steps)
    if not is_adaptive:
        t = sigmas.to(dtype=ode_dtype)
    else:
        t = torch.stack([t_max, t_min]).to(dtype=ode_dtype)

    samples = torch.empty_like(x)

    for i in trange(batch, desc=solver, leave=False):
        if is_adaptive:
            pbar = tqdm(
                total=100,
                desc=f"[tde adaptive] {solver}",
                unit="%",
                bar_format="{desc}: {percentage:.2f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                postfix={"σ": f"{t_max.item():.4f}"},
            )
        else:
            pbar = tqdm(
                total=n_steps - 1,
                desc=f"[tde fixed] {solver}",
                unit="step",
                bar_format="{desc}: {percentage:.2f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
                postfix={"σ": f"{t_max.item():.4f}"},
            )

        with pbar:
            ode = TDEODEFunction(
                model       = model,
                t_min       = t_min.item(),
                t_max       = t_max.item(),
                n_steps     = n_steps,
                is_adaptive = is_adaptive,
                extra_args  = extra_args,
                callback    = callback,
                cfg_denoiser= cfg_denoiser,
                pbar        = pbar,
            )

            try:
                # Separate options for fixed and adaptive step methods
                if is_adaptive:
                    odeint_options = {
                        "min_step":      1e-5,
                        "max_num_steps": max_steps,
                        "dtype":         ode_dtype,
                    }
                else:
                    odeint_options = {
                        "dtype": ode_dtype,
                    }

                result = torchdiffeq.odeint(
                    ode,
                    x[i].to(dtype=ode_dtype),
                    t,
                    rtol   = 10 ** log_rtol,
                    atol   = 10 ** log_atol,
                    method = solver,
                    options= odeint_options,
                )
                samples[i] = result[-1].to(dtype=x.dtype)
                pbar.update(pbar.total - pbar.n)

            except sd_samplers_common.InterruptedException:
                logger.debug("[TDE Sampler] Interrupted (InterruptedException)")
                samples[i] = x[i]

    # Final callback
    if callback is not None:
        callback({
            "x":         samples,
            "i":         n_steps - 1,
            "sigma":     t_min,
            "sigma_hat": t_min,
            "denoised":  samples,
        })

    return samples


# ===========================================================================
# Section 3: reForge Sampler Wrapper Class
# ===========================================================================

class TDEMethodSampler(sd_samplers_common.Sampler):
    """reForge sampler for a single solver."""

    def __init__(self, sd_model, solver: str):
        super().__init__(lambda *a, **k: None)
        self.funcname  = f"tde_{solver}"
        self.solver    = solver

        from modules.sd_samplers_kdiffusion import CFGDenoiserKDiffusion
        self.model_wrap_cfg = CFGDenoiserKDiffusion(self)
        self.model_wrap     = self.model_wrap_cfg.inner_model

    def initialize(self, p) -> dict:
        self.p = p
        self.model_wrap_cfg.p               = p
        self.model_wrap_cfg.mask            = getattr(p, "mask",  None)
        self.model_wrap_cfg.nmask           = getattr(p, "nmask", None)
        self.model_wrap_cfg.step            = 0
        self.model_wrap_cfg.total_steps     = p.steps
        self.model_wrap_cfg.steps           = p.steps
        self.model_wrap_cfg.image_cfg_scale = getattr(p, "image_cfg_scale", None)
        self.eta          = p.eta if p.eta is not None else 0.0
        self.s_min_uncond = getattr(p, "s_min_uncond", 0.0)

        # TorchHijack
        try:
            from modules.sd_samplers_common import TorchHijack
            hijack = TorchHijack(p)
            if opts.sd_sampling == "A1111":
                from k_diff.k_diffusion import sampling as _s
            else:
                from ldm_patched.k_diffusion import sampling as _s
            _s.torch = hijack
            try:
                from ldm_patched.k_diffusion import sampling as _s2
                if _s2 is not _s:
                    _s2.torch = hijack
            except Exception:
                pass
        except Exception:
            pass

        return {}

    def get_sigmas(self, p, steps):
        from modules import sd_schedulers

        discard = (
            self.config is not None
            and self.config.options.get("discard_next_to_last_sigma", False)
        ) or opts.always_discard_next_to_last_sigma

        if discard:
            steps += 1
            p.extra_generation_params["Discard penultimate sigma"] = True

        scheduler_name = (p.hr_scheduler if p.is_hr_pass else p.scheduler) or "Automatic"
        if scheduler_name == "Automatic":
            scheduler_name = self.config.options.get("scheduler", None)

        scheduler = sd_schedulers.schedulers_map.get(scheduler_name)

        m_sigma_min = self.model_wrap.sigmas[0].item()
        m_sigma_max = self.model_wrap.sigmas[-1].item()
        sigma_min, sigma_max = (
            (0.1, 10)
            if getattr(opts, 'use_old_karras_scheduler_sigmas', False)
            else (m_sigma_min, m_sigma_max)
        )

        if p.sampler_noise_scheduler_override:
            sigmas = p.sampler_noise_scheduler_override(steps)
        elif scheduler is None or scheduler.function is None:
            sigmas = self.model_wrap.get_sigmas(steps)
        else:
            kwargs = {"sigma_min": sigma_min, "sigma_max": sigma_max}

            if scheduler.label != "Automatic" and not p.is_hr_pass:
                p.extra_generation_params["Schedule type"] = scheduler.label
            elif scheduler.label != p.extra_generation_params.get("Schedule type"):
                p.extra_generation_params["Hires schedule type"] = scheduler.label

            if opts.sigma_min != 0 and opts.sigma_min != m_sigma_min:
                kwargs["sigma_min"] = opts.sigma_min
            if opts.sigma_max != 0 and opts.sigma_max != m_sigma_max:
                kwargs["sigma_max"] = opts.sigma_max
            if scheduler.default_rho != -1 and opts.rho != 0 and opts.rho != scheduler.default_rho:
                kwargs["rho"] = opts.rho
            if scheduler.need_inner_model:
                kwargs["inner_model"] = self.model_wrap

            sigmas = scheduler.function(n=steps, **kwargs, device=shared.device)

        if discard:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas.cpu()

    def _run(self, p, x, sigmas, solver=None):
        _solver = solver or self.solver

        log_rtol  = _resolve_tde_log_rtol(p)
        log_atol  = _resolve_tde_log_atol(p)
        max_steps = int(getattr(p, "_tde_max_steps", _get(OPT_MAX_STEPS, DEF_MAX_STEPS)))

        if getattr(p, "is_hr_pass", False):
            p.extra_generation_params["TDE hires solver"]   = _solver
            p.extra_generation_params["TDE hires log_rtol"] = log_rtol
            p.extra_generation_params["TDE hires log_atol"] = log_atol
        else:
            p.extra_generation_params["TDE solver"]   = _solver
            p.extra_generation_params["TDE log_rtol"] = log_rtol
            p.extra_generation_params["TDE log_atol"] = log_atol

        return _run_tde_sampler(
            solver      = _solver,
            model       = self.model_wrap_cfg,
            x           = x,
            sigmas      = sigmas,
            extra_args  = self.sampler_extra_args,
            callback    = self.callback_state,
            cfg_denoiser= self.model_wrap_cfg,
            log_rtol    = log_rtol,
            log_atol    = log_atol,
            max_steps   = max_steps,
        )

    def sample(self, p, x, conditioning, unconditional_conditioning,
               steps=None, image_conditioning=None):
        solver = getattr(p, "_tde_txt2img_solver", USE_SAME)

        # Delegate to RK Sampler when TO_RK is selected.
        # Record the UI selection BEFORE delegating: delegated runs never reach
        # _run(), which is where the "TDE solver" key is normally written, so
        # without this the PNG would carry no record of the delegation and
        # paste could not restore the dropdown to "→ RK Sampler".
        if solver == TO_RK:
            p.extra_generation_params["TDE solver"] = solver
            return self._delegate_to_rk(p, x, conditioning, unconditional_conditioning,
                                        steps=steps, image_conditioning=image_conditioning,
                                        is_img2img=False)

        # Delegate to the default sampler when USE_SAME is selected.
        # Recorded for the same reason: the PNG should carry the actual UI
        # selection for every delegation path, not only for solver-run passes.
        if solver == USE_SAME:
            p.extra_generation_params["TDE solver"] = solver
            return self._delegate(p, x, conditioning, unconditional_conditioning,
                                  steps=steps, image_conditioning=image_conditioning,
                                  is_img2img=False)

        try:
            from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
        except ModuleNotFoundError:
            from modules_forge.forge_sampler import sampling_prepare, sampling_cleanup
        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)


        steps  = steps or p.steps
        sigmas = self.get_sigmas(p, steps).to(x.device)

        if opts.sgm_noise_multiplier:
            p.extra_generation_params["SGM noise multiplier"] = True
            x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        else:
            x = x * sigmas[0]

        self.initialize(p)
        self.last_latent = x
        self.sampler_extra_args = {
            "cond":         conditioning,
            "image_cond":   image_conditioning,
            "uncond":       unconditional_conditioning,
            "cond_scale":   p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }

        solver = getattr(p, "_tde_txt2img_solver", self.solver)
        samples = self._run(p, x, sigmas, solver=solver)
        self.add_infotext(p)
        sampling_cleanup(unet_patcher)
        return samples

    def sample_img2img(self, p, x, noise, conditioning, unconditional_conditioning,
                       steps=None, image_conditioning=None):
        solver = getattr(p, "_tde_hr_solver", USE_SAME)

        # Key selection mirrors _run(): the hires pass writes "TDE hires solver",
        # a plain img2img pass writes "TDE solver".
        _solver_key = ("TDE hires solver" if getattr(p, "is_hr_pass", False)
                       else "TDE solver")

        # Delegate to RK Sampler when TO_RK is selected.
        # Record the UI selection BEFORE delegating: delegated runs never reach
        # _run(), so without this the PNG would carry no record of the
        # delegation and paste could not restore the dropdown to "→ RK Sampler".
        if solver == TO_RK:
            p.extra_generation_params[_solver_key] = solver
            return self._delegate_to_rk(p, x, conditioning, unconditional_conditioning,
                                        steps=steps, image_conditioning=image_conditioning,
                                        noise=noise, is_img2img=True)

        # Delegate to the default sampler when USE_SAME is selected.
        # Recorded for the same reason: every delegation path leaves the actual
        # UI selection in the PNG.
        if solver == USE_SAME:
            p.extra_generation_params[_solver_key] = solver
            return self._delegate(p, x, conditioning, unconditional_conditioning,
                                  steps=steps, image_conditioning=image_conditioning,
                                  noise=noise, is_img2img=True)

        try:
            from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
        except ModuleNotFoundError:
            from modules_forge.forge_sampler import sampling_prepare, sampling_cleanup
        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)


        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
        sigmas       = self.get_sigmas(p, steps).to(x.device)
        sigma_sched  = sigmas[steps - t_enc - 1:]

        x  = x.to(noise)
        xi = _flow_aware_noise_injection(x, noise, sigma_sched[0], self.model_wrap)

        if opts.img2img_extra_noise > 0:
            p.extra_generation_params["Extra noise"] = opts.img2img_extra_noise
            from modules.script_callbacks import ExtraNoiseParams, extra_noise_callback
            enp = ExtraNoiseParams(noise, x, xi)
            extra_noise_callback(enp)
            noise = enp.noise
            xi   += noise * opts.img2img_extra_noise

        self.initialize(p)
        self.model_wrap_cfg.init_latent = x
        self.last_latent = x
        self.sampler_extra_args = {
            "cond":         conditioning,
            "image_cond":   image_conditioning,
            "uncond":       unconditional_conditioning,
            "cond_scale":   p.cfg_scale,
            "s_min_uncond": self.s_min_uncond,
        }

        solver = getattr(p, "_tde_hr_solver", self.solver)
        samples = self._run(p, xi, sigma_sched, solver=solver)
        self.add_infotext(p)
        sampling_cleanup(unet_patcher)
        return samples

    def _delegate(self, p, x, conditioning, unconditional_conditioning,
                  steps=None, image_conditioning=None, noise=None, is_img2img=False):
        """Delegate to reForge's default sampler when 'Use same sampler' is selected."""
        from modules import sd_samplers
        fallback_name = getattr(opts, "sampler_name", None) or "Euler"
        sampler = sd_samplers.create_sampler(fallback_name, shared.sd_model)
        if is_img2img:
            return sampler.sample_img2img(
                p, x, noise, conditioning, unconditional_conditioning,
                steps=steps, image_conditioning=image_conditioning
            )
        return sampler.sample(
            p, x, conditioning, unconditional_conditioning,
            steps=steps, image_conditioning=image_conditioning
        )

    def _delegate_to_rk(self, p, x, conditioning, unconditional_conditioning,
                        steps=None, image_conditioning=None, noise=None, is_img2img=False):
        """Delegate to RK Sampler when '→ RK Sampler' is selected."""
        from modules import sd_samplers
        sampler = sd_samplers.create_sampler("RK Sampler", shared.sd_model)
        if is_img2img:
            return sampler.sample_img2img(
                p, x, noise, conditioning, unconditional_conditioning,
                steps=steps, image_conditioning=image_conditioning
            )
        return sampler.sample(
            p, x, conditioning, unconditional_conditioning,
            steps=steps, image_conditioning=image_conditioning
        )


# ===========================================================================
# Section 4: Script UI — Unified Sampler "TDE Sampler"
# ===========================================================================

class TDEScriptSampler(TDEMethodSampler):
    """For the "TDE Sampler" dropdown. Integrates with Script UI method selection."""

    def __init__(self, sd_model):
        super().__init__(sd_model, "euler")
        self.funcname = "tde_script_sampler"


# ===========================================================================
# Section 5: Settings Tab UI
# ===========================================================================

def _on_ui_settings():
    section = ("tde_sampler", "TDE Sampler")

    shared.opts.add_option(OPT_LOG_RTOL, shared.OptionInfo(
        default=DEF_LOG_RTOL,
        label="Log Relative Tolerance (10^x)",
        component=gr.Slider,
        component_args={"minimum": -7.0, "maximum": 0.0, "step": 0.5},
        section=section,
    ).info("Effective for ae_* methods. Smaller = more precise but slower."))

    shared.opts.add_option(OPT_LOG_ATOL, shared.OptionInfo(
        default=DEF_LOG_ATOL,
        label="Log Absolute Tolerance (10^x)",
        component=gr.Slider,
        component_args={"minimum": -7.0, "maximum": 0.0, "step": 0.5},
        section=section,
    ).info("Effective for ae_* methods. Smaller = more precise but slower."))

    shared.opts.add_option(OPT_MAX_STEPS, shared.OptionInfo(
        default=DEF_MAX_STEPS,
        label="Max ODE Steps",
        component=gr.Slider,
        component_args={"minimum": 1, "maximum": 500, "step": 1},
        section=section,
    ).info("Upper limit for adaptive steps. Default for reForge ODE Custom is 250."))


script_callbacks.on_ui_settings(_on_ui_settings)


# ===========================================================================
# Section 6: Sampler Registration
# ===========================================================================

USE_SAME    = "Use same sampler"
TO_RK       = "→ RK Sampler"
SOLVER_NAMES = [USE_SAME, TO_RK] + ALL_SOLVERS


def _register():
    from modules import sd_samplers

    added = 0

    # "TDE Sampler" — unified sampler integrated with Script UI
    tde_script_data = sd_samplers_common.SamplerData(
        name        = "TDE Sampler",
        constructor = lambda sd_model: TDEScriptSampler(sd_model),
        aliases     = ["tde_sampler"],
        options     = {"scheduler": None},
    )
    if not any(s.name == "TDE Sampler" for s in sd_samplers.all_samplers):
        sd_samplers.all_samplers.append(tde_script_data)
        added += 1
    sd_samplers.all_samplers_map["TDE Sampler"] = tde_script_data

    sd_samplers.set_samplers()
    if added > 0:
        logger.warning(
            "[TDE Sampler] Registered %d sampler(s) (torchdiffeq=%s)",
            added, HAS_TORCHDIFFEQ
        )


def _on_model_loaded(sd_model):
    try:
        _register()
        from modules import sd_samplers
        logger.warning(
            "[TDE Sampler] on_model_loaded: TDE Sampler in all_samplers_map = %s",
            "TDE Sampler" in sd_samplers.all_samplers_map
        )
    except Exception:
        import traceback
        logger.error("[TDE Sampler] on_model_loaded error:\n%s", traceback.format_exc())


script_callbacks.on_model_loaded(_on_model_loaded)

try:
    _register()
    logger.warning("[TDE Sampler] Startup registration complete")
except Exception:
    import traceback
    logger.error("[TDE Sampler] Registration error:\n%s", traceback.format_exc())


# ===========================================================================
# Section 7: Script Class — adds UI to the Script pane in the generation tab
# ===========================================================================

try:
    from modules import scripts

    class TDESamplerScript(scripts.Script):

        def title(self):
            return "TDE Sampler"

        def show(self, is_img2img):
            return scripts.AlwaysVisible

        def ui(self, is_img2img):
            from modules.ui_components import InputAccordion
            with InputAccordion(False, label="TDE Sampler") as enabled:
                with gr.Row():
                    txt2img_solver = gr.Dropdown(
                        choices=SOLVER_NAMES,
                        value=USE_SAME,
                        label="txt2img Solver",
                    )
                    hr_solver = gr.Dropdown(
                        choices=SOLVER_NAMES,
                        value=USE_SAME,
                        label="hires.fix Solver",
                        visible=not is_img2img,
                    )
                with gr.Row():
                    txt2img_log_rtol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.5,
                        value=DEF_LOG_RTOL,
                        label="txt2img Log Relative Tolerance (10^x)",
                    )
                    txt2img_log_atol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.5,
                        value=DEF_LOG_ATOL,
                        label="txt2img Log Absolute Tolerance (10^x)",
                    )
                with gr.Row():
                    hires_log_rtol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.5,
                        value=DEF_LOG_RTOL,
                        label="hires.fix Log Relative Tolerance (10^x)",
                        visible=not is_img2img,
                    )
                    hires_log_atol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.5,
                        value=DEF_LOG_ATOL,
                        label="hires.fix Log Absolute Tolerance (10^x)",
                        visible=not is_img2img,
                    )
                with gr.Accordion("Advanced Settings", open=False):
                    max_steps = gr.Slider(
                        minimum=1, maximum=500, step=1,
                        value=_get(OPT_MAX_STEPS, DEF_MAX_STEPS),
                        label="Max ODE Steps",
                    )
                    _tde_max_steps_sliders.append(max_steps)

                # Disable saving/restoring these sliders to ui-config.json.
                # They have no elem_id, so WebUI (modules/ui_loadsave.py)
                # persists their values to ui-config.json keyed by label and,
                # on startup, restores them over the code-defined `value`.
                # As a result, reinstalling the extension keeps stale values
                # instead of the code defaults (DEF_LOG_RTOL / DEF_LOG_ATOL).
                # Setting do_not_save_to_config skips both saving and
                # restoring, so the code `value` is always used. Runtime
                # adjustments are still passed via p._tde_txt2img_log_rtol
                # etc. in process(), so generation is unaffected.
                for _slider in (txt2img_log_rtol, txt2img_log_atol,
                                hires_log_rtol, hires_log_atol,
                                max_steps):
                    _slider.do_not_save_to_config = True

            # PNG infotext round-trip (Send to txt2img / img2img).
            # Keys must match those written in add_infotext(), where solver and
            # tolerances are already recorded per pass ("TDE solver" for txt2img,
            # "TDE hires solver" for hires), so the two sets of controls restore
            # independently.
            #
            # enabled uses a callable: a bare key string can never force the
            # accordion OFF, because paste leaves a component untouched when its
            # key is absent. Returning False on a missing key forces OFF when a
            # PNG generated without TDE Sampler is pasted. Presence of either
            # solver key means it was on.
            #
            # max_steps is intentionally omitted: it lives under Advanced
            # Settings, is not written to infotext, and was never meant to be
            # round-tripped.
            self.infotext_fields = [
                (enabled, lambda d: ("TDE solver" in d)
                                    or ("TDE hires solver" in d)),
                (txt2img_solver,   "TDE solver"),
                (hr_solver,        "TDE hires solver"),
                (txt2img_log_rtol, "TDE log_rtol"),
                (txt2img_log_atol, "TDE log_atol"),
                (hires_log_rtol,   "TDE hires log_rtol"),
                (hires_log_atol,   "TDE hires log_atol"),
            ]

            return [enabled, txt2img_solver, hr_solver,
                    txt2img_log_rtol, txt2img_log_atol,
                    hires_log_rtol, hires_log_atol,
                    max_steps]

        def process(self, p,
                    enabled, txt2img_solver, hr_solver,
                    txt2img_log_rtol, txt2img_log_atol,
                    hires_log_rtol, hires_log_atol,
                    max_steps):
            # Do nothing when disabled
            if not enabled:
                return

            p._tde_txt2img_solver   = txt2img_solver
            p._tde_hr_solver        = hr_solver
            p._tde_txt2img_log_rtol = float(txt2img_log_rtol)
            p._tde_txt2img_log_atol = float(txt2img_log_atol)
            p._tde_hr_log_rtol      = float(hires_log_rtol)
            p._tde_hr_log_atol      = float(hires_log_atol)
            p._tde_max_steps        = int(max_steps)

except ImportError:
    pass


# Force-update sliders to Settings values on page load
# Use demo.load event to work around Gradio 3.x caching issues
def _on_app_started(demo, app):
    if not _tde_max_steps_sliders:
        return
    try:
        def _get_max_steps():
            val = _get(OPT_MAX_STEPS, DEF_MAX_STEPS)
            return [val] * len(_tde_max_steps_sliders)
        with demo:
            demo.load(fn=_get_max_steps, inputs=[], outputs=_tde_max_steps_sliders)
        logger.warning("[TDE Sampler] demo.load registered: %d slider(s)", len(_tde_max_steps_sliders))
    except Exception:
        import traceback
        logger.error("[TDE Sampler] demo.load registration error:\n%s", traceback.format_exc())


script_callbacks.on_app_started(_on_app_started)
