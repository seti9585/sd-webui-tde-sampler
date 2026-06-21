# sd-webui-TDE-Sampler

**EN** | [日本語](#日本語)

ODE sampler extension for Stable Diffusion WebUI (Forge-based),  
powered by [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

Port of [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE) by redhottensors.

> Unlike reForge's built-in ODE Custom, this extension registers as an independent sampler,  
> allowing different solvers to be selected for txt2img and hires.fix separately.

---

## Features

- Registers **9 solvers** as the **"TDE Sampler"** entry in the sampling method dropdown.
- **txt2img and Hires.fix can use different solvers** via the Script accordion.
- Adaptive solvers adjust step size automatically to meet the tolerance targets.
- Fixed-step solvers follow the WebUI step count like standard samplers.
- Supports **Flow Matching models** (Anima / FLUX / SD3) with correct noise injection in Hires.fix.
- Parameters (relative tolerance / absolute tolerance / max steps) can be set per-generation in the Script UI, overriding the Settings tab defaults.
- Generation parameters are embedded in PNG infotext for reproducibility.

---

## Dependency

```bash
pip install torchdiffeq
```

No other packages are needed beyond what is already included in your WebUI.

> This extension relies on the Forge backend sampling functions.  
> It is not available in A1111 (AUTOMATIC1111).

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-TDE-Sampler
```

---

## Usage

### Sampler dropdown

Select **"TDE Sampler"** from the sampling method dropdown.  
When the Script accordion is disabled or set to **"Use same sampler"**, sampling is
delegated to the solver selected in the main sampler dropdown.

### Script accordion

Expand the **"TDE Sampler"** accordion in the script section to configure:

| Control | Description |
|---|---|
| **txt2img Solver** | Solver used for txt2img (and img2img). |
| **hires.fix Solver** | Solver used for the Hires.fix pass only. |
| **Log Relative Tolerance** | `10^x` relative tolerance (rtol) for adaptive solvers. |
| **Log Absolute Tolerance** | `10^x` absolute tolerance (atol) for adaptive solvers. |
| **Max ODE Steps** | Step count cap for adaptive solvers. |

Special values for both solver dropdowns:

- **Use same sampler** — Delegates to the sampler selected in the dropdown.
- **→ RK Sampler** — Routes to RK Sampler for that pass.

---

## ODE Formulation

```
dx/dσ = (x − D(x, σ)) / σ
```

`D(x, σ)` is the denoised latent predicted by the model at noise level σ.  
The probability-flow ODE is integrated from σ_max down to σ_min.

---

## Solvers

### Adaptive Step

Step size is determined automatically to satisfy the relative tolerance (rtol) and
absolute tolerance (atol) targets. The scheduler type and step count are used only
for the sigma range; the actual number of ODE evaluations is controlled by the
tolerances. In CUDA environments, integration is performed in float64 for accurate
error estimation.

| Method | Algorithm | Order | Notes |
|---|---|---|---|
| `dopri5` | Dormand–Prince | 5 | The most widely used adaptive ODE solver. Well-balanced general-purpose choice. |
| `dopri8` | Dormand–Prince | 8 | Higher-order variant. Highest accuracy at highest cost. |
| `bosh3` | Bogacki–Shampine | 3 | Lightweight. Efficient when looser tolerances are acceptable. |
| `fehlberg2` | Fehlberg | 2 | Low order. Fast, but needs tighter tolerances to converge cleanly. |
| `adaptive_heun` | Heun | 2 | Simplest adaptive solver. Like `fehlberg2`, benefits from tighter tolerances. |

### Fixed Step

Fixed-step solvers ignore the tolerances and step exactly along the sigma schedule,
using the WebUI step count directly.

| Method | Algorithm | Order | Notes |
|---|---|---|---|
| `euler` | Euler | 1 | Simplest method. Fastest per step. |
| `midpoint` | Midpoint | 2 | Slightly smoother than Euler. |
| `heun3` | Heun | 3 | Heun's 3rd-order method. |
| `rk4` | Classical Runge–Kutta | 4 | Good quality at moderate step counts. |

---

## Settings

Persistent defaults are available under **Settings → TDE Sampler**.

| Parameter | Default | Range | Description |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | `10^x` rtol for adaptive solvers. Smaller = more precise, slower. Must be ≥ the Log Absolute Tolerance. |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | `10^x` atol for adaptive solvers. Smaller = more precise, slower. Must be ≤ the Log Relative Tolerance. |
| **Max ODE Steps** | 250 | 1 〜 5000 | Failsafe step count cap for adaptive solvers. Has no effect on fixed-step solvers. |

> **Note:** Always set the Log Relative Tolerance greater than or equal to the Log Absolute Tolerance.

---

## Reproducibility

The following parameters are embedded in generated PNG metadata:

```
TDE solver, TDE log_rtol, TDE log_atol
```

---

---

# 日本語

**[English](#sd-webui-tde-sampler)** | 日本語

[torchdiffeq](https://github.com/rtqichen/torchdiffeq) を使った ODE サンプラー拡張機能（Forge 系 WebUI 向け）。

[ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE)（redhottensors 作）を reForge 向けに移植。

> reForge 組み込みの ODE Custom とは独立したサンプラーとして登録されるため、  
> txt2img と hires.fix で異なるソルバーを選択できます。

---

## 特徴

- Sampling method ドロップダウンに **9 種類のソルバー**を **「TDE Sampler」** として登録。
- **txt2img と Hires.fix で異なるソルバー**を Script アコーディオンで選択可能。
- 可変ステップ法は許容誤差の目標値を満たすようステップ幅を自動調整。
- 固定ステップ法は標準サンプラーと同様に WebUI のステップ数に従う。
- Hires.fix での **Flow Matching モデル**（Anima / FLUX / SD3 等）の正しいノイズ注入に対応。
- パラメータ（相対許容誤差 / 絶対許容誤差 / 最大ステップ数）は Script UI で生成ごとに上書き可能（Settings タブのデフォルトより優先）。
- 生成パラメータは PNG の infotext に記録され再現性を保持。

---

## 依存ライブラリ

```bash
pip install torchdiffeq
```

WebUI に既に含まれるもの以外、追加パッケージは不要です。

> 本拡張機能は Forge バックエンドの sampling 関数に依存します。  
> A1111（AUTOMATIC1111）では使用できません。

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-TDE-Sampler
```

---

## 使い方

### Sampler ドロップダウン

Sampling method ドロップダウンから **「TDE Sampler」** を選択します。  
Script アコーディオンが無効、または **「Use same sampler」** の場合は、メインの
サンプラードロップダウンで選択中のソルバーに処理を委譲します。

### Script アコーディオン

Script セクションの **「TDE Sampler」** アコーディオンを展開して設定します。

| 項目 | 説明 |
|---|---|
| **txt2img Solver** | txt2img（および img2img）で使うソルバー。 |
| **hires.fix Solver** | Hires.fix パスでのみ使うソルバー。 |
| **Log Relative Tolerance** | 可変ステップ法の `10^x` 相対許容誤差（rtol）。 |
| **Log Absolute Tolerance** | 可変ステップ法の `10^x` 絶対許容誤差（atol）。 |
| **Max ODE Steps** | 可変ステップ法のステップ数上限。 |

両ソルバードロップダウンの特殊値：

- **Use same sampler** — ドロップダウンで選択中のサンプラーに委譲。
- **→ RK Sampler** — そのパスを RK Sampler に委譲。

---

## ODE の定式化

```
dx/dσ = (x − D(x, σ)) / σ
```

`D(x, σ)` はノイズレベル σ におけるモデルの denoised latent 予測値。  
probability flow ODE を σ_max から σ_min へ降順に積分します。

---

## ソルバー一覧

### 可変ステップ法

ステップ幅は相対許容誤差（rtol）と絶対許容誤差（atol）の目標値を満たすよう自動決定
されます。スケジューラの種類やステップ数は sigma の範囲にのみ使用され、実際の ODE
評価回数は許容誤差で制御されます。CUDA 環境では精度の高い誤差推定のため float64 で
積分を実行します。

| メソッド | アルゴリズム | 次数 | 特徴 |
|---|---|---|---|
| `dopri5` | Dormand–Prince | 5 | 最も広く使われる可変ステップ ODE ソルバー。バランスの取れた汎用ソルバー。 |
| `dopri8` | Dormand–Prince | 8 | 高次版。最高精度だが最も低速。 |
| `bosh3` | Bogacki–Shampine | 3 | 軽量。許容誤差を緩めても許容できる場面で効率的。 |
| `fehlberg2` | Fehlberg | 2 | 低次。高速だが、きれいに収束させるには許容誤差を厳しめにする必要がある。 |
| `adaptive_heun` | Heun | 2 | 最もシンプルな可変ステップソルバー。`fehlberg2` 同様、許容誤差を厳しめにすると安定する。 |

### 固定ステップ法

固定ステップ法は許容誤差を無視し、sigma スケジュールに沿って正確に刻みます。WebUI の
ステップ数をそのまま使用します。

| メソッド | アルゴリズム | 次数 | 特徴 |
|---|---|---|---|
| `euler` | Euler | 1 | 最もシンプル。1 ステップあたりが最速。 |
| `midpoint` | Midpoint | 2 | Euler より若干なめらか。 |
| `heun3` | Heun | 3 | Heun の 3 次法。 |
| `rk4` | 古典的 Runge–Kutta | 4 | 中程度のステップ数で高品質。 |

---

## Settings

**Settings → TDE Sampler** でデフォルト値を永続的に設定できます。

| パラメータ | デフォルト | 範囲 | 説明 |
|---|---|---|---|
| **Log Relative Tolerance** | −3.0 | −7.0 〜 0.0 | 可変ステップソルバーの `10^x` rtol。小さいほど高精度・低速。Log Absolute Tolerance 以上にする。 |
| **Log Absolute Tolerance** | −4.0 | −7.0 〜 0.0 | 可変ステップソルバーの `10^x` atol。小さいほど高精度・低速。Log Relative Tolerance 以下にする。 |
| **Max ODE Steps** | 250 | 1 〜 5000 | 可変ステップソルバーのステップ数上限（フェイルセーフ）。固定ステップ法には影響しない。 |

> **注意：** Log Relative Tolerance は必ず Log Absolute Tolerance 以上にして下さい。

---

## 再現性

生成された PNG のメタデータには以下のパラメータが記録されます。

```
TDE solver, TDE log_rtol, TDE log_atol
```

---

## License

MIT License — Original port: [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE) © redhottensors
