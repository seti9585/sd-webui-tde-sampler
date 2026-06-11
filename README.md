# sd-webui-TDE-Sampler

**EN** | [日本語](#日本語)

ODE sampler extension for Stable Diffusion WebUI (Forge-based),  
powered by [torchdiffeq](https://github.com/rtqichen/torchdiffeq).

Port of [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE) by redhottensors.

> Unlike reForge's built-in ODE Custom, this extension registers as an independent sampler,  
> allowing different solvers to be selected for txt2img and hires.fix separately.

---

## Dependency

```bash
pip install torchdiffeq
```

---

## Installation

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-TDE-Sampler
```

---

## Solvers

| Type | Methods |
|---|---|
| Adaptive | `dopri8` `dopri5` `bosh3` `fehlberg2` `adaptive_heun` |
| Fixed-step | `euler` `midpoint` `rk4` `heun3` |

Adaptive solvers use `rtol` / `atol` to control step size automatically.  
Fixed-step solvers use the WebUI step count directly.

---

## Script UI

Selecting **TDE Sampler** in the Script panel exposes per-generation controls.

| Control | Description |
|---|---|
| txt2img Solver | Solver for the base pass |
| hires.fix Solver | Solver for the hires.fix pass (independent) |
| Log Relative Tolerance | `10^x` rtol for adaptive solvers |
| Log Absolute Tolerance | `10^x` atol for adaptive solvers |
| Max ODE Steps | Upper bound on adaptive step count |

Setting a solver to **`Use same sampler`** falls back to the WebUI's selected sampler.  
Setting it to **`→ RK Sampler`** delegates to the RK Sampler extension.

---

## Settings

Persistent defaults are available under **Settings → TDE Sampler**.

| Setting | Default |
|---|---|
| Log Relative Tolerance | −3.0 |
| Log Absolute Tolerance | −4.0 |
| Max ODE Steps | 250 |

---

## ODE Formulation

```
dx/dσ = (x − D(x, σ)) / σ
```

`D(x, σ)` is the denoised latent predicted by the model at noise level σ.

---

---

# 日本語

**[English](#sd-webui-tde-sampler)** | 日本語

[torchdiffeq](https://github.com/rtqichen/torchdiffeq) を使った ODE サンプラー拡張機能（Forge 系 WebUI 向け）。

[ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE)（redhottensors 作）を reForge 向けに移植。

> reForge 組み込みの ODE Custom とは独立したサンプラーとして登録されるため、  
> txt2img と hires.fix で異なるソルバーを選択できます。

---

## 依存ライブラリ

```bash
pip install torchdiffeq
```

---

## インストール

**Extensions → Install from URL:**

```
https://github.com/seti9585/sd-webui-TDE-Sampler
```

---

## ソルバー一覧

| 種類 | メソッド |
|---|---|
| 適応ステップ | `dopri8` `dopri5` `bosh3` `fehlberg2` `adaptive_heun` |
| 固定ステップ | `euler` `midpoint` `rk4` `heun3` |

適応ステップ法は `rtol` / `atol` に基づいてステップ幅を自動調整します。  
固定ステップ法は WebUI のステップ数をそのまま使用します。

---

## Script UI

Script パネルで **TDE Sampler** を選択すると、生成ごとのパラメータを設定できます。

| 項目 | 説明 |
|---|---|
| txt2img Solver | ベースパスのソルバー |
| hires.fix Solver | hires.fix パスのソルバー（独立して選択可能） |
| Log Relative Tolerance | 適応ステップ法の rtol（`10^x`） |
| Log Absolute Tolerance | 適応ステップ法の atol（`10^x`） |
| Max ODE Steps | 適応ステップの上限数 |

ソルバーを **`Use same sampler`** にすると WebUI で選択中のサンプラーにフォールバックします。  
**`→ RK Sampler`** にすると RK Sampler 拡張機能に処理を委譲します。

---

## Settings

**Settings → TDE Sampler** でデフォルト値を永続的に設定できます。

| 設定項目 | デフォルト値 |
|---|---|
| Log Relative Tolerance | −3.0 |
| Log Absolute Tolerance | −4.0 |
| Max ODE Steps | 250 |

---

## ODE の定式化

```
dx/dσ = (x − D(x, σ)) / σ
```

`D(x, σ)` はノイズレベル σ におけるモデルの denoised latent 予測値。

---

## ライセンス

MIT License — Original port: [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE) © redhottensors
