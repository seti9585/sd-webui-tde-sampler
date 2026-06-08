"""
TDE Sampler — reForge 拡張機能
================================
配置場所: extensions/tde-sampler/scripts/tde_sampler.py

ComfyUI-ODE (https://github.com/redhottensors/ComfyUI-ODE) の
torchdiffeq ベースの ODE サンプラーを reForge に移植。

【reForge 組み込みの ODE Custom との違い】
- reForge の ODE Custom も同じ torchdiffeq を使用しているが、
  この拡張機能は独立したサンプラーとして登録するため、
  txt2img と hires.fix で別々のメソッドを選択可能。
- Script UI で Method・rtol・atol をその場で変更できる。

【移植時の注意（Forge Neo 対応）】
  reForge:  from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
  Forge Neo: from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
  ↑ この1行を差し替えるだけで Forge Neo でも動くはず。

依存:
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
# rk_core と同様、拡張フォルダを sys.path に追加
# ---------------------------------------------------------------------------
_EXT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _EXT_DIR not in sys.path:
    sys.path.insert(0, _EXT_DIR)

# ---------------------------------------------------------------------------
# reForge コアモジュール
# ---------------------------------------------------------------------------
from modules import sd_samplers_common, shared, script_callbacks
from modules.shared import opts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# torchdiffeq チェック
# ---------------------------------------------------------------------------
try:
    import torchdiffeq
    HAS_TORCHDIFFEQ = True
except ModuleNotFoundError:
    HAS_TORCHDIFFEQ = False
    logger.error(
        "[TDE Sampler] torchdiffeq が見つかりません。pip install torchdiffeq を実行してください。"
    )

# ---------------------------------------------------------------------------
# MPS チェック
# ---------------------------------------------------------------------------
HAS_MPS = torch.backends.mps.is_available()


# ===========================================================================
# Section 0: メソッド定義
# ===========================================================================

ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS    = {"euler", "midpoint", "rk4", "heun3"}
ALL_SOLVERS      = sorted([*ADAPTIVE_SOLVERS, *FIXED_SOLVERS])

# opts キー
OPT_LOG_RTOL  = "tde_sampler_log_rtol"
OPT_LOG_ATOL  = "tde_sampler_log_atol"
OPT_MAX_STEPS = "tde_sampler_max_steps"

# デフォルト値（reForge の ODE Custom に合わせる）
DEF_LOG_RTOL  = -2.5
DEF_LOG_ATOL  = -3.5
DEF_MAX_STEPS = 250

# ページロード時の強制更新用スライダー参照リスト
_tde_max_steps_sliders = []


def _get(key, default):
    return getattr(opts, key, default)


# ===========================================================================
# Section 1: ODE 右辺関数（ComfyUI-ODE の ODEFunction を reForge 用に移植）
# ===========================================================================

class TDEODEFunction:
    """
    torchdiffeq.odeint から呼ばれる ODE の右辺関数。
    バッチを1枚ずつ処理する（torchdiffeq の仕様）。

    probability flow ODE:
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
            t : scalar tensor  現在の sigma 値
            y : (C*H*W,) tensor  現在の latent（flatten 済み、1枚分）
        Returns:
            dy/dt : (C*H*W,) tensor
        """
        if t <= 1e-5:
            return torch.zeros_like(y)

        # torchdiffeq は1枚ずつ処理するので unsqueeze でバッチ次元を追加
        denoised = self.model(
            y.unsqueeze(0),
            t.unsqueeze(0),
            **self.extra_args
        )
        return (y - denoised.squeeze(0)) / t

    def callback_step(self, t0, y0, dt):
        """固定ステップ: 各ステップ後に呼ばれる。"""
        if self.is_adaptive:
            return

        self._fire_callback(t0, y0, self.step)

        if self.pbar is not None:
            self.pbar.update(1)
            self.pbar.set_postfix({"σ": f"{t0.item():.4f}"})
            self.pbar.refresh()

        # cfg_denoiser.step を更新
        if self.cfg_denoiser is not None and hasattr(self.cfg_denoiser, "step"):
            total = getattr(self.cfg_denoiser, "total_steps", None)
            if total is not None:
                self.cfg_denoiser.step = min(self.step, total - 1)
            else:
                self.cfg_denoiser.step = self.step

        self.step += 1

    def callback_accept_step(self, t0, y0, dt):
        """適応ステップ: 採択ステップ後に呼ばれる。"""
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

        # cfg_denoiser.step を更新
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
# Section 2: コアサンプリング関数
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
    torchdiffeq.odeint を使って ODE を解く。
    ComfyUI-ODE の ODESampler.__call__() に相当。
    バッチを1枚ずつ処理する。
    """
    is_adaptive = solver in ADAPTIVE_SOLVERS
    t_max   = sigmas.max()
    t_min   = sigmas.min()
    n_steps = len(sigmas)
    batch   = x.shape[0]

    # torchdiffeq の dtype 設定
    ode_dtype = torch.float32 if HAS_MPS else torch.float64

    # sigma スケジュール（固定ステップ）or [t_max, t_min]（適応ステップ）
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
                # 固定ステップ法と適応ステップ法でオプションを分ける
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
                logger.debug("[TDE Sampler] 中断されました（InterruptedException）")
                samples[i] = x[i]

    # 最終コールバック
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
# Section 3: reForge Sampler ラッパークラス
# ===========================================================================

class TDEMethodSampler(sd_samplers_common.Sampler):
    """1つの solver に対応する reForge サンプラー。"""

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
            if opts.use_old_karras_scheduler_sigmas
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

        log_rtol  = getattr(p, "_tde_log_rtol",  _get(OPT_LOG_RTOL,  DEF_LOG_RTOL))
        log_atol  = getattr(p, "_tde_log_atol",  _get(OPT_LOG_ATOL,  DEF_LOG_ATOL))
        max_steps = int(getattr(p, "_tde_max_steps", _get(OPT_MAX_STEPS, DEF_MAX_STEPS)))

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

        # 「→ RK Sampler」が選ばれているとき RK Sampler に委譲
        if solver == TO_RK:
            return self._delegate_to_rk(p, x, conditioning, unconditional_conditioning,
                                        steps=steps, image_conditioning=image_conditioning,
                                        is_img2img=False)

        # 「Use same sampler」のとき reForge のデフォルトサンプラーに委譲
        if solver == USE_SAME:
            return self._delegate(p, x, conditioning, unconditional_conditioning,
                                  steps=steps, image_conditioning=image_conditioning,
                                  is_img2img=False)

        from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)

        self.model_wrap.log_sigmas = self.model_wrap.log_sigmas.to(x.device)
        self.model_wrap.sigmas     = self.model_wrap.sigmas.to(x.device)

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

        # 「→ RK Sampler」が選ばれているとき RK Sampler に委譲
        if solver == TO_RK:
            return self._delegate_to_rk(p, x, conditioning, unconditional_conditioning,
                                        steps=steps, image_conditioning=image_conditioning,
                                        noise=noise, is_img2img=True)

        # 「Use same sampler」のとき reForge のデフォルトサンプラーに委譲
        if solver == USE_SAME:
            return self._delegate(p, x, conditioning, unconditional_conditioning,
                                  steps=steps, image_conditioning=image_conditioning,
                                  noise=noise, is_img2img=True)

        from backend.sampling.sampling_function import sampling_prepare, sampling_cleanup
        unet_patcher = self.model_wrap.inner_model.forge_objects.unet
        sampling_prepare(unet_patcher, x=x)

        self.model_wrap.log_sigmas = self.model_wrap.log_sigmas.to(x.device)
        self.model_wrap.sigmas     = self.model_wrap.sigmas.to(x.device)

        steps, t_enc = sd_samplers_common.setup_img2img_steps(p, steps)
        sigmas       = self.get_sigmas(p, steps).to(x.device)
        sigma_sched  = sigmas[steps - t_enc - 1:]

        x  = x.to(noise)
        xi = x + noise * sigma_sched[0]

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
        """「Use same sampler」のとき reForge のデフォルトサンプラーに委譲。"""
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
        """「→ RK Sampler」のとき RK Sampler に委譲。"""
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
# Section 4: Script UI — 統合サンプラー「TDE Sampler」
# ===========================================================================

class TDEScriptSampler(TDEMethodSampler):
    """「TDE Sampler」ドロップダウン用。Script UI のメソッド選択と連携する。"""

    def __init__(self, sd_model):
        super().__init__(sd_model, "euler")
        self.funcname = "tde_script_sampler"


# ===========================================================================
# Section 5: Settings タブ UI
# ===========================================================================

def _on_ui_settings():
    section = ("tde_sampler", "TDE Sampler")

    shared.opts.add_option(OPT_LOG_RTOL, shared.OptionInfo(
        default=DEF_LOG_RTOL,
        label="Log Relative Tolerance (10^x)",
        component=gr.Slider,
        component_args={"minimum": -7.0, "maximum": 0.0, "step": 0.5},
        section=section,
    ).info("ae_* メソッドで有効。小さいほど精密・低速。"))

    shared.opts.add_option(OPT_LOG_ATOL, shared.OptionInfo(
        default=DEF_LOG_ATOL,
        label="Log Absolute Tolerance (10^x)",
        component=gr.Slider,
        component_args={"minimum": -7.0, "maximum": 0.0, "step": 0.5},
        section=section,
    ).info("ae_* メソッドで有効。小さいほど精密・低速。"))

    shared.opts.add_option(OPT_MAX_STEPS, shared.OptionInfo(
        default=DEF_MAX_STEPS,
        label="Max ODE Steps",
        component=gr.Slider,
        component_args={"minimum": 1, "maximum": 5000, "step": 1},
        section=section,
    ).info("適応ステップの上限。reForge ODE Custom のデフォルトは 250。"))


script_callbacks.on_ui_settings(_on_ui_settings)


# ===========================================================================
# Section 6: サンプラー登録
# ===========================================================================

USE_SAME    = "Use same sampler"
TO_RK       = "→ RK Sampler"
SOLVER_NAMES = [USE_SAME, TO_RK] + ALL_SOLVERS


def _register():
    from modules import sd_samplers

    added = 0

    # 「TDE Sampler」— Script UI と連携する統合サンプラー
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
            "[TDE Sampler] %d サンプラーを登録しました (torchdiffeq=%s)",
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
        logger.error("[TDE Sampler] on_model_loaded エラー:\n%s", traceback.format_exc())


script_callbacks.on_model_loaded(_on_model_loaded)

try:
    _register()
    logger.warning("[TDE Sampler] 起動時登録完了")
except Exception:
    import traceback
    logger.error("[TDE Sampler] 登録エラー:\n%s", traceback.format_exc())


# ===========================================================================
# Section 7: Script クラス — 生成タブの Script ペインに UI を追加
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
                    log_rtol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.5,
                        value=DEF_LOG_RTOL,
                        label="Log Relative Tolerance (10^x)",
                    )
                    log_atol = gr.Slider(
                        minimum=-7.0, maximum=0.0, step=0.5,
                        value=DEF_LOG_ATOL,
                        label="Log Absolute Tolerance (10^x)",
                    )
                with gr.Accordion("拡張設定", open=False):
                    max_steps = gr.Slider(
                        minimum=1, maximum=5000, step=1,
                        value=_get(OPT_MAX_STEPS, DEF_MAX_STEPS),
                        label="Max ODE Steps",
                    )
                    _tde_max_steps_sliders.append(max_steps)

            return [enabled, txt2img_solver, hr_solver, log_rtol, log_atol, max_steps]

        def process(self, p,
                    enabled, txt2img_solver, hr_solver, log_rtol, log_atol, max_steps):
            # 無効のときは何もしない
            if not enabled:
                return

            p._tde_txt2img_solver = txt2img_solver
            p._tde_hr_solver      = hr_solver
            p._tde_log_rtol       = float(log_rtol)
            p._tde_log_atol       = float(log_atol)
            p._tde_max_steps      = int(max_steps)

except ImportError:
    pass


# ページロード時にスライダーを Settings の値で強制更新する
# Gradio 3.x のキャッシュ問題を回避するために demo.load イベントを使う
def _on_app_started(demo, app):
    if not _tde_max_steps_sliders:
        return
    try:
        def _get_max_steps():
            val = _get(OPT_MAX_STEPS, DEF_MAX_STEPS)
            return [val] * len(_tde_max_steps_sliders)
        with demo:
            demo.load(fn=_get_max_steps, inputs=[], outputs=_tde_max_steps_sliders)
        logger.warning("[TDE Sampler] demo.load 登録完了: %d スライダー", len(_tde_max_steps_sliders))
    except Exception:
        import traceback
        logger.error("[TDE Sampler] demo.load 登録エラー:\n%s", traceback.format_exc())


script_callbacks.on_app_started(_on_app_started)
