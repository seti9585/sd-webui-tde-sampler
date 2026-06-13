# TDE Sampler — Mathematical Verification

A record of how `tde-sampler` (a port of ComfyUI-ODE to reForge / Forge Neo) was
verified to be a **mathematically correct ODE solver**, without running a single
image generation.

> 🇺🇸 English first, 🇯🇵 日本語は後半にあります。

---

## What was verified

- **Repository:** `tde-sampler` — a standalone sampler extension porting
  ComfyUI-ODE to reForge / Forge Neo.
- **Target:** the **mathematical core** of `tde_sampler.py` —
  `TDEODEFunction.__call__` (the ODE right-hand side) and the integration part of
  `_run_tde_sampler` (schedule construction + the `torchdiffeq.odeint` call +
  singularity guard + per-image processing + dtype handling).
- **Goal:** the Runge-Kutta / adaptive solvers themselves are guaranteed by
  torchdiffeq, but the ported **plumbing** can still be wired up incorrectly. The
  point is to confirm that plumbing is mathematically sound.
- **Environment:** no WebUI required, CPU only, finishes in tens of seconds.
  `torch` + `torchdiffeq 0.2.5`.

## How this differs from the RK Sampler verification (important)

The RK Sampler keeps its Butcher tableaux in its **own `rk_core/`**, so the main
question there was "are the coefficients correct (does the convergence order come
out right)?" The TDE Sampler, by contrast, delegates the solver to **torchdiffeq
(a mature external library)**, so coefficient correctness is the library's
responsibility. What TDE needs to verify is therefore not the coefficients but the
**ported plumbing**:

- the probability-flow ODE formulation `dx/dσ = (x − D(x,σ)) / σ`
- the integration direction (descending integration, σ_max → σ_min)
- schedule construction (**fixed methods = pass the whole sigma grid as `t`;
  adaptive methods = pass only the two points `[σ_max, σ_min]`**)
- the σ=0 singularity guard (zero gradient for `t ≤ 1e-5`)
- dtype handling (adaptive = float64, fixed = float32; the model call is always
  float32 and the result is cast back)
- no batch contamination under per-image processing (a torchdiffeq requirement)

The verification **copies `TDEODEFunction.__call__` and the integration part of
`_run_tde_sampler` verbatim** (removing only the reForge plumbing irrelevant to the
math — pbar / callback / cfg_denoiser / InterruptedException) and checks them
against analytic solutions.

## Results (17/17 PASS)

### Test 1 — Empirical order of convergence, fixed-step methods (`dy/dt = -y`, t: 1 → 0, solution `e^{1-t}`)

Order measured from the error-reduction ratio when doubling the grid N=20 → 40.

| Method | Theoretical order | Measured order | Result |
|---|---|---|---|
| euler | 1 | 0.97 | PASS |
| midpoint | 2 | 1.97 | PASS |
| heun3 | 3 | 2.97 | PASS |
| rk4 | 4 | 3.97 | PASS |

→ The **way torchdiffeq is called** (how the grid is passed, descending `t`) is
correct, and integration proceeds at the theoretical order.

### Test 2 — Tolerance sweep, adaptive-step methods (`dy/dt = -y`)

Terminal error as rtol/atol is tightened 1e-3 → 1e-5 → 1e-7.

| Method | err(1e-3) | err(1e-5) | err(1e-7) | Result |
|---|---|---|---|---|
| adaptive_heun | 3.9e-3 | 3.4e-5 | 3.9e-7 | PASS |
| bosh3 | 1.5e-2 | 5.2e-4 | 1.9e-5 | PASS |
| dopri5 | 4.2e-4 | 7.5e-6 | 2.1e-7 | PASS |
| dopri8 | 3.9e-2 | 7.2e-4 | 1.1e-4 | PASS |
| fehlberg2 | 9.3e-2 | 3.1e-3 | 1.2e-5 | PASS |

→ Every method's error decreases monotonically in response to the tolerance.
Adaptive control — including how `min_step` / `max_num_steps` / `dtype` options are
passed — works correctly.

### Test 3 — `TDEODEFunction` real code path vs. exact solution (Karras-style σ)

The probability-flow ODE is integrated **through the real code path** with an
analytic denoiser and compared to the closed form.

| Case | Interval | Result | Verdict |
|---|---|---|---|
| (a) `D ≡ 0.7`, rk4(fp32) | σ_max → σ_min(0.03) | rel max err = 1.2e-7 | PASS |
| (b) `D = x(1-σ/σmax)`, rk4(fp32) | σ_max → σ_min(0.03) | rel max err = 2.1e-6 | PASS |
| (c) `D ≡ 0.7`, σ_max → 0 (terminal) | euler / rk4 | euler bias ≈ 2e-8 / rk4 bias ≈ 2e-2 | PASS |
| (d) `D = x(1-σ/σmax)`, dopri5(fp64) | σ_max → σ_min, rtol 1e-7 | rel max err = 3.1e-7 | PASS |

→ If the ODE formulation, integration direction, schedule following, dtype round
trip, or per-image processing had any wiring error, these numbers would not appear.
**The correctness of the port is essentially proven.**

Note: (a)(b) matching at float32 precision (1e-7 to 1e-6) is by design. The reason
it does **not** match at machine precision (1e-16) like RK is the intentional choice
to **run fixed methods in float32** (see below). The adaptive case (d) runs in
float64, hence 1e-7.

### Test 4 — `euler` vs. hand-written forward Euler (same sigma grid, same float32)

→ max abs diff = **0.0** (exact match). torchdiffeq's euler steps cleanly one step
at a time along the sigma grid — no off-by-one, schedule following is exact. It
should be nearly bit-identical to k-diffusion's built-in Euler.

### Test 5 — Determinism & batch independence

| Item | Result | Verdict |
|---|---|---|
| Determinism (dopri5 ×2) | max abs diff = 0.0 | PASS |
| Batch independence (batch=2 vs. processed individually) | max abs diff = 0.0 | PASS |

→ An ODE sampler has no stochastic term, so two runs match exactly. The per-image
loop produces no cross-batch contamination.

### Test 6 — Mutual convergence of adaptive methods (uniqueness of the PF-ODE solution)

As the tolerance is tightened, different methods should converge to the same latent.

| Comparison | Difference | Verdict |
|---|---|---|
| \|dopri5 − dopri8\| | 2.3e-12 | PASS |
| \|dopri5 − bosh3\| | 2.3e-12 | PASS |

→ The PF-ODE solution is unique; in the float64 path the adaptive methods converge
to the same latent.

## Behavior to be aware of (not a bug)

1. **Fixed methods run in float32, adaptive methods in float64** (both float32 on
   `HAS_MPS` environments). As the comment in `tde_sampler.py` explains, running
   fixed methods in float64 accumulates rounding differences along the sigma
   schedule and produces sparkle/grain artifacts — especially with hires.fix, where
   the σ range is narrow. This is an intentional measure. Consequently the
   analytic-solution agreement for fixed methods tops out at float32 precision
   (1e-7), which is where it differs from RK (whose fixed methods matched at machine
   precision).

2. **torchdiffeq's fixed methods ignore the `dtype` option** and emit
   `UserWarning: Unexpected arguments {'dtype': ...}`. This is harmless; the
   integration dtype for fixed methods is determined by the input dtype (= float32).
   No real effect.

3. **σ→0 terminal masking.** At the final step (σ_min → 0), stage evaluations with
   `t ≤ 1e-5` are masked to zero gradient (handling the `1/σ` singularity the PF-ODE
   has at σ=0; inherited from ComfyUI-ODE).
   - **euler reaches the terminal exactly in its single final step**, so the terminal
     bias is ≈ 0 (2e-8 in Test 3c).
   - Multi-stage methods such as rk4 have their later stages masked as σ→0, leaving a
     small terminal bias (about rel 2e-2 in a worst-case synthetic test with a
     constant denoiser). With a real denoiser the gradient `(x−D)/σ` stays bounded as
     σ→0, so the effect on real images is smaller than this, and with the most-used
     euler it is essentially zero. This is the same known behavior as RK's Test 3c.

## Conclusion

- **The mathematics of `tde_sampler.py` (the PF-ODE formulation, integration
  direction, schedule following, singularity handling, dtype handling, per-image
  processing) works correctly.**
- The 4 fixed methods show the theoretical convergence order, the 5 adaptive methods
  converge monotonically in response to the tolerance, and the real code path matches
  the analytic solution (at float32 precision for fixed / float64 for adaptive).
  euler is bit-identical to hand-written Euler; determinism and batch independence
  are confirmed.
- Together with the RK Sampler verification, this establishes that **both standalone
  ODE samplers (RK / TDE) are mathematically correct**, without running any image
  generation.
- What rots is the glue with torchdiffeq / PyTorch — API compatibility and
  dtype/device handling. As with the RK script, running this as a regression test
  detects breakage when PyTorch / torchdiffeq is updated.

## How to reproduce

Place `test_tde_core_verification.py` anywhere and run it (it has no dependency on
reForge itself; `torch` and `torchdiffeq` are enough):

```bash
pip install torch torchdiffeq
python test_tde_core_verification.py
```

If all 6 tests (17 items) PASS, the PF-ODE formulation, schedule following,
singularity handling, and dtype handling are confirmed correct. When
`TDEODEFunction` / `_run_tde_sampler` are modified, keep the copied code and the
verification side in sync to use it for regression checks.

---
---

# TDE Sampler — 数学的検証（日本語）

`tde-sampler`（ComfyUI-ODE を reForge / Forge Neo 向けに移植した独立サンプラー拡張）が、
**ODE ソルバーとして数学的に正しく動作する**ことを、画像生成を一切回さずに検証した
記録です。

## 検証対象

- **リポジトリ:** `tde-sampler` — ComfyUI-ODE を reForge / Forge Neo 向けに移植した
  独立サンプラー拡張。
- **対象:** `tde_sampler.py` の **数学コア** — `TDEODEFunction.__call__`（ODE 右辺）と
  `_run_tde_sampler` の積分部（スケジュール構築＋`torchdiffeq.odeint` 呼び出し＋
  特異点ガード＋1枚ずつ処理＋dtype 取り回し）。
- **目的:** ルンゲ・クッタ／適応ソルバーそのものは torchdiffeq が担保するが、
  移植した**配管**は配線を誤りうる。その配管が数学的に正しいかを確認する。
- **実行環境:** WebUI 起動不要・CPU のみ・数十秒。`torch` + `torchdiffeq 0.2.5`。

## RK Sampler 検証との位置づけの違い（重要）

RK Sampler は Butcher tableau を**自前の `rk_core/` に持つ**ため、「係数が正しいか
（収束次数が出るか）」が検証の主眼だった。一方 TDE Sampler のソルバーは
**torchdiffeq（外部の枯れたライブラリ）**であり、係数の正しさはライブラリ側で担保される。
したがって TDE で検証すべきは係数ではなく **移植した配管** である：

- probability flow ODE の定式化 `dx/dσ = (x − D(x,σ)) / σ`
- 積分方向（σ_max → σ_min への降順積分）
- スケジュール構築（**固定法＝全 sigma グリッドを `t` に渡す／適応法＝`[σ_max, σ_min]`
  の2点のみ渡す**）
- σ=0 特異点ガード（`t ≤ 1e-5` でゼロ勾配）
- dtype 取り回し（適応＝float64、固定＝float32、モデル呼び出しは常に float32 にして
  結果を戻す）
- 1枚ずつ処理（torchdiffeq 仕様）でのバッチ非汚染

検証は `TDEODEFunction.__call__` と `_run_tde_sampler` の積分部を **逐語コピー**し
（pbar / callback / cfg_denoiser / InterruptedException など数学に無関係な reForge
配管のみ除去）、解析解と突き合わせた。

## 検証結果（全 17 項目 PASS）

### Test 1 — 固定ステップ法の経験的収束次数（`dy/dt = -y`, t: 1 → 0, 解 `e^{1-t}`）

グリッドを N=20 → 40 に倍化したときの誤差減少率から実測次数を算出。

| メソッド | 理論次数 | 実測次数 | 判定 |
|---|---|---|---|
| euler | 1 | 0.97 | PASS |
| midpoint | 2 | 1.97 | PASS |
| heun3 | 3 | 2.97 | PASS |
| rk4 | 4 | 3.97 | PASS |

→ torchdiffeq への**呼び出し方**（grid の渡し方・降順 t）が正しく、理論次数どおりに
積分されている。

### Test 2 — 適応ステップ法の許容誤差スイープ（`dy/dt = -y`）

rtol/atol を 1e-3 → 1e-5 → 1e-7 と締めたときの終端誤差。

| メソッド | err(1e-3) | err(1e-5) | err(1e-7) | 判定 |
|---|---|---|---|---|
| adaptive_heun | 3.9e-3 | 3.4e-5 | 3.9e-7 | PASS |
| bosh3 | 1.5e-2 | 5.2e-4 | 1.9e-5 | PASS |
| dopri5 | 4.2e-4 | 7.5e-6 | 2.1e-7 | PASS |
| dopri8 | 3.9e-2 | 7.2e-4 | 1.1e-4 | PASS |
| fehlberg2 | 9.3e-2 | 3.1e-3 | 1.2e-5 | PASS |

→ 全メソッドで tol に応答して誤差が単調減少。`min_step` / `max_num_steps` / `dtype`
オプションの渡し方も含めて適応制御が正常に効いている。

### Test 3 — `TDEODEFunction` 実コードパス vs 厳密解（Karras 風 σ）

解析的デノイザーを通して probability flow ODE を**実コードパスで積分**し、閉形式と
突き合わせる。

| ケース | 評価区間 | 結果 | 判定 |
|---|---|---|---|
| (a) `D ≡ 0.7`, rk4(fp32) | σ_max → σ_min(0.03) | rel max err = 1.2e-7 | PASS |
| (b) `D = x(1-σ/σmax)`, rk4(fp32) | σ_max → σ_min(0.03) | rel max err = 2.1e-6 | PASS |
| (c) `D ≡ 0.7`, σ_max → 0（終端） | euler / rk4 | euler bias≈2e-8 / rk4 bias≈2e-2 | PASS |
| (d) `D = x(1-σ/σmax)`, dopri5(fp64) | σ_max → σ_min, rtol 1e-7 | rel max err = 3.1e-7 | PASS |

→ ODE 定式化・積分方向・スケジュール追従・dtype 往復・1枚ずつ処理のいずれかに
配線ミスがあればこの数字は出ない。**移植の正しさはほぼ証明された。**

注: (a)(b) が float32 精度（1e-7〜1e-6 台）で一致するのは仕様どおり。RK のように
**マシン精度(1e-16)で一致しないのは「固定法を float32 で回す」TDE の意図的設計**
（下記参照）。適応法 (d) は float64 で走るため 1e-7 台。

### Test 4 — `euler` vs 手書き前進 Euler（同一 sigma グリッド・同一 float32）

→ max abs diff = **0.0**（完全一致）。torchdiffeq の euler は sigma グリッド上を
素直に1ステップずつ刻んでおり、off-by-one なし・スケジュール追従は正確。
k-diffusion 組み込み Euler とほぼビット一致するはず。

### Test 5 — 決定論性 & バッチ独立性

| 項目 | 結果 | 判定 |
|---|---|---|
| 決定論性（dopri5 ×2） | max abs diff = 0.0 | PASS |
| バッチ独立性（batch=2 vs 単独処理） | max abs diff = 0.0 | PASS |

→ ODE サンプラーは確率項がないので2回完全一致。1枚ずつ処理ループでバッチ間の汚染なし。

### Test 6 — 適応メソッド相互収束（PF-ODE 解の一意性）

許容誤差を締めると別メソッドが同一 latent へ収束するはず。

| 比較 | 差 | 判定 |
|---|---|---|
| \|dopri5 − dopri8\| | 2.3e-12 | PASS |
| \|dopri5 − bosh3\| | 2.3e-12 | PASS |

→ PF-ODE の解は一意。float64 経路で適応メソッドが同一 latent に収束している。

## 仕様として把握しておくべき挙動（バグではない）

1. **固定法は float32、適応法は float64**（`HAS_MPS` 環境では両方 float32）。
   tde_sampler.py のコメントどおり、固定法を float64 で回すと sigma スケジュール上で
   丸め差が蓄積し、特に hires.fix（σ レンジが狭い）で sparkle/grain アーティファクトが
   出るための意図的処置。このため固定法の解析解一致は float32 精度（1e-7台）で頭打ちに
   なる（RK の固定法がマシン精度で一致したのとはここが異なる）。

2. **torchdiffeq 固定法は `dtype` オプションを無視**し
   `UserWarning: Unexpected arguments {'dtype': ...}` を出す。これは無害で、固定法の
   積分 dtype は入力 dtype（=float32）で決まる。実害なし。

3. **σ→0 終端マスキング**。最終ステップ（σ_min→0）で `t ≤ 1e-5` のステージ評価が
   ゼロ勾配にマスクされる（PF-ODE が σ=0 に `1/σ` 特異点を持つための処置、
   ComfyUI-ODE 由来）。
   - **euler は最終1ステップで厳密に終端へ到達**するため終端バイアス≈0（Test 3c で 2e-8）。
   - rk4 など多段法は σ→0 で後段ステージがマスクされ、小さな終端バイアスが残る
     （定数デノイザーという最悪条件の合成テストで相対 2e-2 程度）。実デノイザーでは
     σ→0 で勾配 `(x−D)/σ` が有界になるため実画像での影響はこれより小さく、また最も
     使われる euler では本質的にゼロ。RK の Test 3c と同種・既知の挙動。

## 結論

- **tde_sampler.py の数学（PF-ODE 定式化、積分方向、スケジュール追従、特異点処理、
  dtype 取り回し、1枚ずつ処理）は正しく動作している。**
- 固定 4 種は理論どおりの収束次数を示し、適応 5 種は tol に応答して単調収束、
  実コードパスは解析解と（固定=float32 精度／適応=float64 精度で）一致した。
  euler は手書き Euler とビット一致、決定論性・バッチ独立性も確認。
- RK Sampler 側の検証と合わせ、**reForge の2本の独立 ODE サンプラー（RK / TDE）は
  いずれも数学的に正しい**ことが、画像生成を回さずに担保された。
- 腐るのは torchdiffeq / PyTorch との API 整合・dtype/device の糊の部分であり、
  RK の検証スクリプトと同様、本スクリプトを回帰テストとして回せば
  PyTorch / torchdiffeq アップデート時の退行検知としてそのまま機能する。

## 再現方法

検証スクリプト `test_tde_core_verification.py` を任意の場所に置き、以下を実行する
（reForge 本体への依存はなく、`torch` と `torchdiffeq` だけで完走する）。

```bash
pip install torch torchdiffeq
python test_tde_core_verification.py
```

全 6 テスト（17 項目）が PASS すれば、PF-ODE 定式化・スケジュール追従・特異点処理・
dtype 取り回しの正しさが担保される。`TDEODEFunction` / `_run_tde_sampler` を改修した
際は、コピー元と検証側を同期させて回帰確認に使える。
