# sd-webui-TDE-Sampler

English | [日本語](#日本語)

TDE Sampler is a sampler extension for reForge / Forge Neo that generates images using numerical methods for solving ODEs (ordinary differential equations). It uses [torchdiffeq](https://github.com/rtqichen/torchdiffeq) and provides 14 solvers.

It is designed to use ODE methods in both txt2img and Hires.fix, with a separate choice of method for each pass. You can also use TDE Sampler for only one pass alongside another sampler, or combine RK Sampler and TDE Sampler across the two passes.

[Installation and tested environments](#installation-and-tested-environments) · [Usage](#usage) · [Adjusting computation and accuracy](#adjusting-computation-and-accuracy) · [Method reference](#method-reference)

## Installation and tested environments

In Extensions → Install from URL, enter:

```text
https://github.com/seti9585/sd-webui-tde-sampler
```

The dependency is `torchdiffeq`. If needed, install it in the Python environment used by your WebUI:

```bash
pip install torchdiffeq
```

Restart the WebUI after installation. This extension uses Forge backend sampling functions and does not support A1111 (AUTOMATIC1111). In reForge, it registers separately from the built-in ODE Custom.

Tested environments:

- reForge, Python 3.10.11, PyTorch 2.9.0+cu128
- Forge Neo 2.29.2, Python 3.13.12, PyTorch 2.13.0+cu130

## Compatibility with Spectrum extensions

When using U-Net models, [Forge Neo's built-in Spectrum (`sd_forge_spectrum`)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo/extensions-builtin/sd_forge_spectrum) and hirorohi03's [sd-forge-spectrum-faithful](https://github.com/hirorohi03/sd-forge-spectrum-faithful) are not designed for higher-order sampling methods that evaluate the U-Net multiple times within one solver step. Using them with these methods can stop generation with an error.

These Spectrum extensions manage forecasting progress by counting U-Net calls, without grouping the internal stages of a solver step. In multi-stage ODE methods, one U-Net call is not one solver step, so their progress and forecast-history handling do not match the sampler's integration.

If you want to use Spectrum-style acceleration alongside this sampler, try my [sd-webui-ChebyCast](https://github.com/seti9585/sd-webui-ChebyCast) instead. Disable the built-in Spectrum and spectrum-faithful extensions before enabling ChebyCast. It distinguishes solver steps from individual U-Net calls and supports fixed-step multi-stage methods. Use `Stage grouping = auto` for TDE / RK Sampler. ChebyCast does not support adaptive ODE methods.

## Usage

### Choose the sampler, then its ODE method

The WebUI sampler dropdowns select which sampler runs. The Script controls select which ODE method TDE Sampler uses.

1. Select `TDE Sampler` in Sampling method. To use it for Hires.fix too, select `TDE Sampler` there as well, or let Hires.fix use the same sampler as txt2img.
2. Enable the `TDE Sampler` accordion in the Script section.
3. Choose an ODE method in `txt2img Solver` and `hires.fix Solver` for each pass you use.
4. Adjust the WebUI step count for a fixed-step method, or the tolerances for an adaptive-step method, then generate.

Selecting a method in the Script section does not change the WebUI sampler selection.

### Combine txt2img and Hires.fix samplers

| Combination | WebUI txt2img sampler | WebUI Hires.fix sampler | Script controls to configure |
|---|---|---|---|
| TDE Sampler for both passes | `TDE Sampler` | `TDE Sampler` or same as txt2img | Choose the solver for each pass |
| TDE Sampler for txt2img only | `TDE Sampler` | Another sampler | Choose the txt2img solver |
| TDE Sampler for Hires.fix only | Another sampler | `TDE Sampler` | Choose the Hires.fix solver |
| RK for txt2img, TDE for Hires.fix | `RK Sampler` | `TDE Sampler` | RK's txt2img Method and TDE's hires.fix Solver |

To combine RK and TDE, install both extensions and enable both Script accordions. The reverse combination—TDE for txt2img and RK for Hires.fix—also works.

### Special dropdown choices

| Choice | Behavior |
|---|---|
| `Use same sampler` | Delegates to the default sampler instead of using this extension's ODE methods |
| `→ RK Sampler` | Hands this pass to RK Sampler, using that extension's method and settings for the same pass |

The fallback for `Use same sampler` is the WebUI `sampler_name` setting, or Euler if that setting is absent. It does not automatically read the txt2img or Hires.fix sampler dropdown. To combine this extension with another sampler, select the samplers explicitly as shown in the table.

When using `→ RK Sampler`, enable the receiving extension's Script accordion and select a concrete method for that pass. Do not create a circular route from RK to TDE and back to RK for the same pass.

## Adjusting computation and accuracy

### Fixed and adaptive steps

| Type | What to adjust | Behavior |
|---|---|---|
| Fixed-step | WebUI step count and scheduler for each pass | Follows the supplied sigma schedule |
| Adaptive-step | Relative and absolute tolerances | Adjusts step size using an error estimate |

For fixed-step methods, the step count determines how finely the calculation is divided. Tolerances and Max ODE Steps are not used. Different methods perform different numbers of model evaluations per step, so equal step counts do not imply equal generation times.

Adaptive methods do not use the WebUI step count as their integration step count. They take the sigma range from the WebUI settings, then choose their steps using the tolerances.

### Reading the tolerances

txt2img and Hires.fix each have their own Log Relative Tolerance and Log Absolute Tolerance controls. The Script values apply to the current generation.

- Log Relative Tolerance: `rtol = 10^x`, an error target relative to the size of the values being integrated.
- Log Absolute Tolerance: `atol = 10^x`, an absolute error target that also applies near zero.

For example, `−3` means `0.001`, and `−4` means `0.0001`. More negative values set stricter tolerances and usually require more computation. These control numerical accuracy, not the visual quality of an image directly. Set Log Relative Tolerance greater than or equal to Log Absolute Tolerance.

The tolerance sliders are greyed out for fixed-step methods, `Use same sampler`, and `→ RK Sampler`, because those choices do not use this extension's tolerances. Values are preserved when you switch back to an adaptive method.

### Advanced settings and defaults

Persistent defaults are available under Settings → TDE Sampler. The Script's Advanced Settings are shared by txt2img and Hires.fix.

| Control | Default | Range | Purpose |
|---|---|---|---|
| Log Relative Tolerance | −3.0 | −7.0 to 0.0 | Relative tolerance for adaptive methods |
| Log Absolute Tolerance | −4.0 | −7.0 to 0.0 | Absolute tolerance for adaptive methods |
| Max ODE Steps | 250 | 1 to 500 | Step cap for adaptive methods |

If an adaptive method reaches the step cap (`max_num_steps exceeded`), loosen the tolerances or raise Max ODE Steps. Strict tolerances can exhaust the cap, particularly in the narrow sigma range used by Hires.fix.

## Method reference

“Order” describes a mathematical accuracy property of the numerical method. A higher order does not directly imply a better-looking image.

### Adaptive-step

| Method | Algorithm | Order |
|---|---|---|
| `dopri5` | Dormand–Prince | 5 |
| `dopri8` | Dormand–Prince | 8 |
| `bosh3` | Bogacki–Shampine | 3 |
| `fehlberg2` | Fehlberg | 2 |
| `adaptive_heun` | Heun | 2 |

### Fixed-step

| Method | Algorithm | Order |
|---|---|---|
| `euler` | Euler | 1 |
| `midpoint` | Midpoint | 2 |
| `heun3` | Heun | 3 |
| `ralston3` | Ralston | 3 |
| `wray3` | van der Houwen / Wray | 3 |
| `ssprk3` | Strong Stability Preserving RK | 3 |
| `kutta4` | Classical Runge–Kutta | 4 |
| `kutta_38th4` | Runge–Kutta 3/8-rule | 4 |
| `ralston4` | Ralston | 4 |

### Older images labeled `rk4`

torchdiffeq's built-in `rk4` uses Kutta's 3/8-rule. This extension distinguishes classical RK4 as `kutta4` and the 3/8-rule as `kutta_38th4`. To reproduce older images labeled `TDE solver: rk4`, select `kutta_38th4`.

## Technical details

### ODE formulation

```text
dx/dσ = (x − D(x, σ)) / σ
```

`x` is the current latent, `σ` is the noise level, and `D(x, σ)` is the denoised latent predicted by the model. The probability-flow ODE is integrated from higher to lower noise levels.

Adaptive methods integrate in float64 in CUDA environments. Fixed-step methods use float32 to match standard k-diffusion samplers. Inputs to the denoising model are cast to float32.

TDE Sampler also supports Hires.fix noise injection for Flow Matching models such as Anima, FLUX, and SD3.

### Settings recorded in PNGs

The method and tolerances are written to PNG infotext separately for txt2img and Hires.fix.

```text
TDE solver, TDE log_rtol, TDE log_atol
TDE hires solver, TDE hires log_rtol, TDE hires log_atol
```

Max ODE Steps is not recorded. When regenerating under the same conditions, also match the advanced settings that are not recorded.

---

# 日本語

[English](#sd-webui-tde-sampler)

TDE Sampler は、ODE（常微分方程式）の数値解法を利用して画像を生成する、reForge / Forge Neo 向けのサンプラー拡張機能です。[torchdiffeq](https://github.com/rtqichen/torchdiffeq) を使い、14種類のソルバーから解法を選べます。

txt2img と Hires.fix の両方で ODE を使い、それぞれ異なる解法を選べる設計です。片方だけを TDE Sampler にして他のサンプラーと組み合わせたり、RK Sampler と TDE Sampler を両段階で使い分けたりすることもできます。

[導入と動作確認環境](#導入と動作確認環境) · [使い方](#使い方) · [計算量と精度の調整](#計算量と精度の調整) · [解法一覧](#解法一覧)

## 導入と動作確認環境

Extensions → Install from URL に次のURLを入力します。

```text
https://github.com/seti9585/sd-webui-tde-sampler
```

依存ライブラリは `torchdiffeq` です。必要な場合は、WebUI が使う Python 環境でインストールしてください。

```bash
pip install torchdiffeq
```

導入後は WebUI を再起動してください。Forge バックエンドの sampling 関数を使うため、A1111（AUTOMATIC1111）では使用できません。reForge では、組み込みの ODE Custom とは別のサンプラーとして登録されます。

動作確認環境：

- reForge、Python 3.10.11、PyTorch 2.9.0+cu128
- Forge Neo 2.29.2、Python 3.13.12、PyTorch 2.13.0+cu130

## Spectrum 拡張機能との併用

U-Net モデルでは、[Forge Neo 内蔵の Spectrum（`sd_forge_spectrum`）](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo/extensions-builtin/sd_forge_spectrum) と、ひろろひ氏（hirorohi03）の [sd-forge-spectrum-faithful](https://github.com/hirorohi03/sd-forge-spectrum-faithful) は、1つの積分ステップ内で U-Net を複数回呼び出す高次の Sampling Method を想定していません。これらのメソッドと併用すると、生成がエラー停止することがあります。

これらの Spectrum 拡張機能は、U-Net の呼び出し回数をもとに予測の進行を管理し、1ステップ内の複数の内部段階（stage）をまとめて扱いません。多段の ODE 解法では「U-Net 呼び出し1回」と「積分ステップ1回」が一致しないため、進行や予測履歴の管理がサンプラーの計算と合わなくなります。

スペクトラム拡張機能を併用する場合は、拙作の [sd-webui-ChebyCast](https://github.com/seti9585/sd-webui-ChebyCast) を試してみてください。内蔵 Spectrum と spectrum-faithful を無効にし、代わりに ChebyCast を有効にします。ChebyCast は積分ステップと個々の U-Net 呼び出しを分けて扱い、固定ステップの多段法に対応しています。TDE / RK Sampler では `Stage grouping = auto` を使ってください。可変ステップの ODE 解法には対応していません。

## 使い方

### サンプラーと解法を選ぶ

WebUI のサンプラー欄では「どのサンプラーを使うか」、Script 欄では「TDE Sampler がどの解法を使うか」を設定します。

1. Sampling method で `TDE Sampler` を選びます。Hires.fix にも使う場合は、Hires.fix のサンプラーも `TDE Sampler` にするか、txt2img と同じサンプラーを使う設定にします。
2. Script 欄の `TDE Sampler` アコーディオンを有効にします。
3. `txt2img Solver` と `hires.fix Solver` で、それぞれ使う解法を選びます。
4. 固定ステップ法なら WebUI のステップ数、可変ステップ法なら許容誤差を調整して生成します。

Script 欄で解法を選ぶだけでは、WebUI のサンプラー選択は切り替わりません。

### txt2img と Hires.fix を使い分ける

| 使いたい組み合わせ | WebUI の txt2img サンプラー | WebUI の Hires.fix サンプラー | Script 欄で設定する解法 |
|---|---|---|---|
| 両方 TDE Sampler | `TDE Sampler` | `TDE Sampler` または txt2img と同じ | txt2img／Hires.fix のソルバーをそれぞれ選ぶ |
| txt2img だけ TDE Sampler | `TDE Sampler` | 他のサンプラー | txt2img のソルバーを選ぶ |
| Hires.fix だけ TDE Sampler | 他のサンプラー | `TDE Sampler` | Hires.fix のソルバーを選ぶ |
| txt2img は RK、Hires.fix は TDE | `RK Sampler` | `TDE Sampler` | RK の txt2img Method と TDE の hires.fix Solver を選ぶ |

RK と TDE を組み合わせる場合は、両拡張機能を導入し、両方の Script アコーディオンを有効にします。上の表の逆に、txt2img を TDE、Hires.fix を RK にすることもできます。

### 解法ドロップダウンの特殊な選択肢

| 選択肢 | 動作 |
|---|---|
| `Use same sampler` | この拡張機能の ODE 解法を使わず、既定のサンプラーへ処理を渡す |
| `→ RK Sampler` | その段階の処理を RK Sampler に渡し、渡した先の同じ段階の解法と設定を使う |

`Use same sampler` の委譲先は WebUI の `sampler_name` 設定で、設定がなければ Euler です。txt2img／Hires.fix のサンプラー欄を自動的に参照して切り替える選択肢ではありません。他のサンプラーと組み合わせる場合は、上の表のように WebUI の各サンプラー欄で選ぶと明確です。

`→ RK Sampler` を使う場合は、渡した先の Script アコーディオンも有効にし、該当する段階で具体的な解法を選んでください。同じ段階で RK → TDE → RK と相互に渡す設定にはしないでください。

## 計算量と精度の調整

### 固定ステップ法と可変ステップ法

| 種類 | 調整する項目 | 動作 |
|---|---|---|
| 固定ステップ法 | WebUI の txt2img／Hires.fix のステップ数とスケジューラ | 指定した sigma スケジュールに沿って計算する |
| 可変ステップ法 | 相対・絶対許容誤差 | 推定誤差に応じてステップ幅を調整する |

固定ステップ法では、ステップ数が計算の細かさを決めます。許容誤差と Max ODE Steps は使いません。解法ごとに1ステップで行うモデル評価の回数が違うため、同じステップ数でも生成時間は異なります。

可変ステップ法では、WebUI のステップ数をそのまま積分ステップ数として使いません。sigma の範囲は WebUI の設定から取得し、その範囲をどう刻むかは許容誤差で決めます。

### 許容誤差の読み方

txt2img と Hires.fix にはそれぞれ、Log Relative Tolerance と Log Absolute Tolerance があります。Script 欄の値は、その生成の設定として使われます。

- Log Relative Tolerance：相対許容誤差 `rtol = 10^x`。値の大きさに応じた誤差の基準。
- Log Absolute Tolerance：絶対許容誤差 `atol = 10^x`。値がゼロに近いときにも使う誤差の基準。

例えば `−3` は `0.001`、`−4` は `0.0001` です。負の値を小さくするほど許容誤差が厳しくなり、通常は計算量が増えます。ODE の計算精度を指定する値であり、画像の見栄えを直接保証する値ではありません。Log Relative Tolerance は Log Absolute Tolerance 以上に設定してください。

固定ステップ法、`Use same sampler`、`→ RK Sampler` を選ぶと、こちらの許容誤差スライダーは使われないためグレーアウトします。値は保持され、可変ステップ法に戻すと再び使えます。

### 詳細設定と既定値

Settings → TDE Sampler で既定値を設定できます。Script 欄の Advanced Settings は txt2img と Hires.fix で共通です。

| 項目 | 既定値 | 範囲 | 用途 |
|---|---|---|---|
| Log Relative Tolerance | −3.0 | −7.0 ～ 0.0 | 可変ステップ法の相対許容誤差 |
| Log Absolute Tolerance | −4.0 | −7.0 ～ 0.0 | 可変ステップ法の絶対許容誤差 |
| Max ODE Steps | 250 | 1 ～ 500 | 可変ステップ法のステップ数上限 |

可変ステップ法がステップ数上限（`max_num_steps exceeded`）で停止する場合は、許容誤差を緩めるか、Max ODE Steps を増やしてください。特に Hires.fix の狭い sigma 範囲では、厳しい許容誤差で上限に達することがあります。

## 解法一覧

「次数」は数値解法の精度に関する数学的な性質です。高い次数が、そのまま画像の見栄えの良さを意味するわけではありません。

### 可変ステップ法

| メソッド | アルゴリズム | 次数 |
|---|---|---|
| `dopri5` | Dormand–Prince | 5 |
| `dopri8` | Dormand–Prince | 8 |
| `bosh3` | Bogacki–Shampine | 3 |
| `fehlberg2` | Fehlberg | 2 |
| `adaptive_heun` | Heun | 2 |

### 固定ステップ法

| メソッド | アルゴリズム | 次数 |
|---|---|---|
| `euler` | Euler | 1 |
| `midpoint` | Midpoint | 2 |
| `heun3` | Heun | 3 |
| `ralston3` | Ralston | 3 |
| `wray3` | van der Houwen / Wray | 3 |
| `ssprk3` | Strong Stability Preserving RK | 3 |
| `kutta4` | 古典的 Runge–Kutta | 4 |
| `kutta_38th4` | Runge–Kutta 3/8 則 | 4 |
| `ralston4` | Ralston | 4 |

### 旧バージョンの `rk4` について

torchdiffeq 組み込みの `rk4` は Kutta の3/8則です。本拡張機能では、古典的 RK4 を `kutta4`、3/8則を `kutta_38th4` と区別しています。旧バージョンの `TDE solver: rk4` を再現する場合は、`kutta_38th4` を選んでください。

## 技術的な詳細

### ODE の定式化

```text
dx/dσ = (x − D(x, σ)) / σ
```

`x` は計算中の latent、`σ` はノイズレベル、`D(x, σ)` はモデルが予測した denoised latent です。この probability-flow ODE を、高いノイズレベルから低いノイズレベルへ積分します。

可変ステップ法は CUDA 環境では float64 で積分し、固定ステップ法は標準の k-diffusion サンプラーに合わせて float32 を使います。モデルへの入力は float32 に揃えます。

TDE Sampler は Flow Matching モデル（Anima / FLUX / SD3 等）の Hires.fix におけるノイズ注入にも対応しています。

### PNG に記録する設定

使用した解法と許容誤差を txt2img／Hires.fix 別に PNG の infotext に記録します。

```text
TDE solver, TDE log_rtol, TDE log_atol
TDE hires solver, TDE hires log_rtol, TDE hires log_atol
```

Max ODE Steps は記録されません。同じ条件で再生成する場合は、記録されない詳細設定も揃えてください。

## License / ライセンス

MIT License — Original port: [ComfyUI-ODE](https://github.com/redhottensors/ComfyUI-ODE) © redhottensors

The fixed-step Runge–Kutta tableaux (`kutta4`, `kutta_38th4`, `ralston3`, `ralston4`, `wray3`, `ssprk3`) match those of [ComfyUI-RK-Sampler](https://github.com/memmaptensor/ComfyUI-RK-Sampler) © memmaptensor, so each is the torchdiffeq-path counterpart of the corresponding RK Sampler method.

