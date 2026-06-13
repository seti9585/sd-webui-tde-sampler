#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDE Sampler 数学的検証スクリプト
================================
WebUI 起動不要・CPU のみ。torch + torchdiffeq があれば数十秒で完走する。

検証対象は tde_sampler.py の「数学コア」のみ:
  - TDEODEFunction.__call__   : probability flow ODE 右辺 dx/dσ = (x - D(x,σ))/σ
  - _run_tde_sampler の積分部 : スケジュール構築 + torchdiffeq.odeint 呼び出し
                                + 特異点ガード + 1枚ずつ処理 + dtype 取り回し

下記 CORE セクションは tde_sampler.py から「逐語コピー」した（pbar / callback /
cfg_denoiser / InterruptedException など reForge 配管のみ除去）。数学に関わる
分岐(ガード・dtype 往復・スケジュール構築・積分方向)は一切改変していない。

RK Sampler との違い:
  RK は Butcher tableau を自前(rk_core)で持つので「係数の正しさ」を検証した。
  TDE のソルバーは torchdiffeq(外部の枯れたライブラリ)なので、検証すべきは
  係数ではなく「配管」: ODE 定式化 / 積分方向 / スケジュール / 特異点処理 /
  dtype / 1枚ずつ処理 が正しく組まれているか。
"""

import math
import warnings
import torch
import torchdiffeq

# 固定ステップ法は 'dtype' オプションを無視して警告を出すが、これは
# tde_sampler.py が意図的に固定法を float32 で回す設計に由来する正常動作。
warnings.filterwarnings("ignore", message=".*Unexpected arguments.*")

torch.manual_seed(0)
DEVICE = "cpu"

ADAPTIVE_SOLVERS = {"dopri8", "dopri5", "bosh3", "fehlberg2", "adaptive_heun"}
FIXED_SOLVERS    = {"euler", "midpoint", "rk4", "heun3"}

# このマシンに MPS は無い → tde_sampler.py の HAS_MPS=False 経路と同じ
HAS_MPS = False


# ===========================================================================
# CORE: tde_sampler.py からの逐語コピー(配管のみ除去)
# ===========================================================================

class TDEODEFunction:
    """tde_sampler.py Section 1 の __call__ を逐語コピー(pbar/callback等は除去)。"""

    def __init__(self, model, extra_args=None):
        self.model      = model
        self.extra_args = extra_args or {}

    def __call__(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if t <= 1e-5:
            return torch.zeros_like(y)

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


def run_tde_core(solver, model, x, sigmas, log_rtol=-3.0, log_atol=-4.0,
                 max_steps=250):
    """tde_sampler.py Section 2 _run_tde_sampler の積分部を逐語コピー
    (pbar/callback/InterruptedException 除去)。dtype 分岐・スケジュール構築・
    odeint 呼び出し・1枚ずつ処理・最終 reshape は原文どおり。"""
    is_adaptive = solver in ADAPTIVE_SOLVERS
    t_max   = sigmas.max()
    t_min   = sigmas.min()
    n_steps = len(sigmas)
    batch   = x.shape[0]

    if HAS_MPS:
        ode_dtype = torch.float32
    elif is_adaptive:
        ode_dtype = torch.float64
    else:
        ode_dtype = torch.float32

    if not is_adaptive:
        t = sigmas.to(dtype=ode_dtype)
    else:
        t = torch.stack([t_max, t_min]).to(dtype=ode_dtype)

    samples = torch.empty_like(x)

    for i in range(batch):
        ode = TDEODEFunction(model=model)

        if is_adaptive:
            odeint_options = {"min_step": 1e-5, "max_num_steps": max_steps,
                              "dtype": ode_dtype}
        else:
            odeint_options = {"dtype": ode_dtype}

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

    return samples


# ===========================================================================
# 解析的デノイザー(RK 検証と同じ2種。flow 系にもそのまま通用)
#   (a) D ≡ c        → 厳密解 x(σ) = c + (x0 - c)·σ/σ0
#   (b) D = x(1-σ/σm)→ 厳密解 x(σ) = x0·exp((σ - σ0)/σm)
# ===========================================================================

def make_const_denoiser(c):
    def model(y, sigma, **kw):
        return torch.full_like(y, float(c))
    return model

def make_linear_denoiser(sigma_max):
    def model(y, sigma, **kw):
        s = sigma.view(-1, *([1] * (y.dim() - 1)))
        return y * (1.0 - s / sigma_max)
    return model


def karras_sigmas(n, sigma_min, sigma_max, rho=7.0, to_zero=False):
    ramp = torch.linspace(0, 1, n)
    min_inv = sigma_min ** (1 / rho)
    max_inv = sigma_max ** (1 / rho)
    sig = (max_inv + ramp * (min_inv - max_inv)) ** rho
    if to_zero:
        sig = torch.cat([sig, torch.zeros(1)])
    return sig.to(torch.float64)


# ===========================================================================
def banner(s): print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)
def line(s):   print(s)

results = []  # (name, passed)


# ---------------------------------------------------------------------------
# Test 1 — torchdiffeq 固定ステップ法の経験的収束次数 (dy/dt = -y, t:1→0)
#   ソルバー自体は torchdiffeq だが「我々の呼び出し方」と「次数」を確認。
# ---------------------------------------------------------------------------
def test1_fixed_order():
    banner("Test 1 — 固定ステップ法の経験的収束次数  (dy/dt = -y, 解 e^{1-t})")
    f = lambda t, y: -y
    y0 = torch.tensor([1.0], dtype=torch.float64)   # y(1) = 1
    exact = math.exp(1.0)  # 解 y(t)=e^{1-t} → y(0)=e^{+1}

    theory = {"euler": 1, "midpoint": 2, "heun3": 3, "rk4": 4}
    line(f"{'method':<10} {'理論':>4} {'実測':>6}  判定")
    for m, p in theory.items():
        errs = []
        for N in (20, 40):
            t = torch.linspace(1.0, 0.0, N + 1, dtype=torch.float64)
            ys = torchdiffeq.odeint(f, y0, t, method=m)
            errs.append(abs(ys[-1].item() - exact))
        order = math.log(errs[0] / errs[1]) / math.log(2.0)
        ok = abs(order - p) < 0.4
        results.append((f"Test1/{m}", ok))
        line(f"{m:<10} {p:>4} {order:>6.2f}  {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# Test 2 — 適応ステップ法の許容誤差スイープ (dy/dt = -y)
# ---------------------------------------------------------------------------
def test2_adaptive_tol():
    banner("Test 2 — 適応ステップ法の許容誤差スイープ  (dy/dt = -y)")
    f = lambda t, y: -y
    y0 = torch.tensor([1.0], dtype=torch.float64)
    t  = torch.tensor([1.0, 0.0], dtype=torch.float64)
    exact = math.exp(1.0)  # y(0)=e^{+1}

    line(f"{'method':<16} {'err(1e-3)':>10} {'err(1e-5)':>10} {'err(1e-7)':>10}  判定")
    for m in sorted(ADAPTIVE_SOLVERS):
        errs = {}
        for tol in (1e-3, 1e-5, 1e-7):
            ys = torchdiffeq.odeint(f, y0, t, rtol=tol, atol=tol, method=m,
                                    options={"max_num_steps": 100000})
            errs[tol] = abs(ys[-1].item() - exact)
        # tol を締めて誤差が(丸め床まで)単調非増加、かつ全体に小さければ PASS
        ok = (errs[1e-3] < 1e-1
              and errs[1e-3] >= errs[1e-5] * 0.5
              and errs[1e-5] >= errs[1e-7] * 0.5)
        results.append((f"Test2/{m}", ok))
        line(f"{m:<16} {errs[1e-3]:>10.1e} {errs[1e-5]:>10.1e} {errs[1e-7]:>10.1e}"
             f"  {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# Test 3 — 実コードパス(run_tde_core) vs 厳密解
#   probability flow ODE を解析的デノイザー経由で積分し閉形式と突き合わせ。
# ---------------------------------------------------------------------------
def test3_real_path():
    banner("Test 3 — 実コードパス(run_tde_core) vs 厳密解")
    sm_max, sm_min = 14.6, 0.03
    sig_smooth = karras_sigmas(25, sm_min, sm_max)            # σmax→0.03(滑らか域)
    sig_to0    = karras_sigmas(25, sm_min, sm_max, to_zero=True)  # …→0(最終ジャンプ込み)
    s0 = sig_smooth[0]

    shape = (3, 8, 8)
    x0 = torch.randn(1, *shape, dtype=torch.float64) * sm_max  # σmax 相当のノイズ

    line(f"{'ケース':<42} {'指標':>16}   判定")

    # (a) D ≡ 0.7, 滑らか域, 固定 rk4(float32 経路) → float32 精度で一致
    c = 0.7
    out = run_tde_core("rk4", make_const_denoiser(c), x0, sig_smooth)
    exact = c + (x0 - c) * (sig_smooth[-1] / s0)
    rel = ((out - exact).abs() / exact.abs().clamp_min(1e-3)).max().item()
    ok = rel < 1e-5
    results.append(("Test3a", ok))
    line(f"{'(a) D≡0.7  rk4(fp32)  σmax→0.03':<42} {('rel max '+f'{rel:.1e}'):>16}   {'PASS' if ok else 'FAIL'}")

    # (b) D = x(1-σ/σmax), 滑らか域, 固定 rk4(float32) → 相対誤差で評価
    out = run_tde_core("rk4", make_linear_denoiser(sm_max), x0, sig_smooth)
    exact = x0 * torch.exp((sig_smooth[-1] - s0) / sm_max)
    rel = ((out - exact).abs() / exact.abs().clamp_min(1e-3)).max().item()
    ok = rel < 1e-4
    results.append(("Test3b", ok))
    line(f"{'(b) D=x(1-σ/σm)  rk4(fp32)  σmax→0.03':<42} {('rel max '+f'{rel:.1e}'):>16}   {'PASS' if ok else 'FAIL'}")

    # (c) D ≡ 0.7, σmax→0(最終ジャンプ込み) → 終端マスキングの定量化(既知仕様)
    #     euler は最終1ステップで厳密に c に到達するので終端バイアス≈0。
    #     rk4 は σ→0 で一部ステージがゼロ勾配にマスクされ小さなバイアスが残る。
    out_e = run_tde_core("euler", make_const_denoiser(c), x0, sig_to0)
    bias_e = ((out_e - c).abs() / abs(c)).max().item()
    out_r = run_tde_core("rk4", make_const_denoiser(c), x0, sig_to0)
    bias_r = ((out_r - c).abs() / abs(c)).max().item()
    ok = bias_e < 1e-5 and bias_r < 5e-2   # euler≈0, rk4 は有界に小さい
    results.append(("Test3c", ok))
    line(f"{'(c) D≡0.7  σmax→0  euler/rk4 終端':<42} {('e'+f'{bias_e:.0e}'+' r'+f'{bias_r:.0e}'):>16}   {'PASS' if ok else 'FAIL'}")

    # (d) 適応 dopri5(fp64 経路, dtype 往復を通る) vs 厳密解
    out = run_tde_core("dopri5", make_linear_denoiser(sm_max), x0, sig_smooth,
                       log_rtol=-7, log_atol=-8)
    exact = x0 * torch.exp((sig_smooth[-1] - s0) / sm_max)
    rel = ((out - exact).abs() / exact.abs().clamp_min(1e-3)).max().item()
    ok = rel < 1e-5
    results.append(("Test3d", ok))
    line(f"{'(d) D=x(1-σ/σm)  dopri5(fp64)  rtol1e-7':<42} {('rel max '+f'{rel:.1e}'):>16}   {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# Test 4 — TDE euler vs 手書き前進 Euler(同一 sigma グリッド)
#   torchdiffeq euler が sigma グリッド上を素直に刻んでいるかの確認。
# ---------------------------------------------------------------------------
def test4_euler_match():
    banner("Test 4 — TDE euler vs 手書き前進 Euler(同一 sigma グリッド)")
    sm_max, sm_min = 14.6, 0.03
    sig = karras_sigmas(25, sm_min, sm_max)
    shape = (3, 8, 8)
    x0 = torch.randn(1, *shape, dtype=torch.float64) * sm_max
    model = make_linear_denoiser(sm_max)

    # 実コードパス(euler は float32 で走る)
    out = run_tde_core("euler", model, x0, sig)

    # 手書き前進 Euler(降順グリッド)を「同じ float32」で:
    #   x += (σ_{i+1}-σ_i)·(x - D)/σ_i
    x = x0.to(torch.float32).clone()
    sig32 = sig.to(torch.float32)
    for i in range(len(sig32) - 1):
        s = sig32[i]
        if s <= 1e-5:
            d = torch.zeros_like(x)
        else:
            D = model(x, s.unsqueeze(0))
            d = (x - D) / s
        x = x + (sig32[i + 1] - sig32[i]) * d
    x = x.to(out.dtype)

    diff = (out - x).abs().max().item()
    ok = diff == 0.0
    results.append(("Test4", ok))
    line(f"max abs diff = {diff:.3e}   {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------------------
# Test 5 — 決定論性 & バッチ独立性
# ---------------------------------------------------------------------------
def test5_determinism_batch():
    banner("Test 5 — 決定論性 & バッチ独立性(1枚ずつ処理の汚染なし)")
    sm_max, sm_min = 14.6, 0.03
    sig = karras_sigmas(20, sm_min, sm_max)
    model = make_linear_denoiser(sm_max)
    s0 = sig[0]

    # 決定論性: 同設定2回で完全一致
    x0 = torch.randn(1, 3, 8, 8, dtype=torch.float64) * sm_max
    a = run_tde_core("dopri5", model, x0, sig, log_rtol=-6, log_atol=-7)
    b = run_tde_core("dopri5", model, x0, sig, log_rtol=-6, log_atol=-7)
    d1 = (a - b).abs().max().item()
    ok1 = d1 == 0.0
    results.append(("Test5/determinism", ok1))
    line(f"決定論性 (dopri5 ×2)            max abs diff = {d1:.3e}   {'PASS' if ok1 else 'FAIL'}")

    # バッチ独立性: batch=2 をまとめて処理 → 各々を単独処理と比較
    xb = torch.randn(2, 3, 8, 8, dtype=torch.float64) * sm_max
    out_batch = run_tde_core("rk4", model, xb, sig)
    out0 = run_tde_core("rk4", model, xb[0:1], sig)
    out1 = run_tde_core("rk4", model, xb[1:2], sig)
    d2 = max((out_batch[0:1] - out0).abs().max().item(),
             (out_batch[1:2] - out1).abs().max().item())
    ok2 = d2 == 0.0
    results.append(("Test5/batch", ok2))
    line(f"バッチ独立性 (batch=2 vs 単独)  max abs diff = {d2:.3e}   {'PASS' if ok2 else 'FAIL'}")


# ---------------------------------------------------------------------------
# Test 6 — 適応メソッドの相互収束(PF-ODE 解の一意性)
#   許容誤差を締めると別メソッドが同一 latent に収束するはず。
# ---------------------------------------------------------------------------
def test6_adaptive_agreement():
    banner("Test 6 — 適応メソッド相互収束(dopri5 vs dopri8 vs bosh3)")
    sm_max, sm_min = 14.6, 0.03
    sig = karras_sigmas(20, sm_min, sm_max)
    model = make_const_denoiser(0.3)
    x0 = torch.randn(1, 3, 8, 8, dtype=torch.float64) * sm_max

    refs = {}
    for m in ("dopri5", "dopri8", "bosh3"):
        refs[m] = run_tde_core(m, model, x0, sig, log_rtol=-9, log_atol=-10)
    d_58 = (refs["dopri5"] - refs["dopri8"]).abs().max().item()
    d_5b = (refs["dopri5"] - refs["bosh3"]).abs().max().item()
    ok = d_58 < 1e-6 and d_5b < 1e-6
    results.append(("Test6", ok))
    line(f"|dopri5 - dopri8| = {d_58:.1e},  |dopri5 - bosh3| = {d_5b:.1e}"
         f"   {'PASS' if ok else 'FAIL'}")


# ===========================================================================
if __name__ == "__main__":
    print("TDE Sampler 数学的検証  (torch %s / torchdiffeq %s)"
          % (torch.__version__, torchdiffeq.__version__))
    test1_fixed_order()
    test2_adaptive_tol()
    test3_real_path()
    test4_euler_match()
    test5_determinism_batch()
    test6_adaptive_agreement()

    banner("総合判定")
    n_pass = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n{n_pass}/{len(results)} PASS")
    print("ALL PASS ✅" if n_pass == len(results) else "SOME FAILED ❌")
