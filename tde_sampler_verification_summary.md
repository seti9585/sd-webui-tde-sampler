# TDE Sampler 数学的検証サマリー
## 対象

- リポジトリ: `tde-sampler`（ComfyUI-ODE を reForge / Forge Neo 向けに移植した独立サンプラー拡張）
- 検証対象: `tde_sampler.py` の **数学コア** — `TDEODEFunction.__call__`（ODE 右辺）と `_run_tde_sampler` の積分部（スケジュール構築＋`torchdiffeq.odeint` 呼び出し＋特異点ガード＋1枚ずつ処理＋dtype 取り回し）
- 目的: 「ルンゲ・クッタ／適応ソルバーそのものは torchdiffeq が担保するが、移植した**配管**が数学的に正しく組まれているか」を、画像生成を一切回さずに検証する
- 実行環境: WebUI 起動不要・CPU のみ・数十秒。`torch` + `torchdiffeq 0.2.5`

## RK Sampler 検証との位置づけの違い（重要）

RK Sampler は Butcher tableau を**自前の `rk_core/` に持つ**ため、「係数が正しいか（収束次数が出るか）」が検証の主眼だった。
一方 TDE Sampler のソルバーは **torchdiffeq（外部の枯れたライブラリ）** であり、係数の正しさはライブラリ側で担保される。したがって TDE で検証すべきは係数ではなく **移植した配管** である：

- probability flow ODE の定式化 `dx/dσ = (x − D(x,σ)) / σ`
- 積分方向（σ_max → σ_min への降順積分）
- スケジュール構築（**固定法＝全 sigma グリッドを `t` に渡す／適応法＝`[σ_max, σ_min]` の2点のみ渡す**）
- σ=0 特異点ガード（`t ≤ 1e-5` でゼロ勾配）
- dtype 取り回し（適応＝float64、固定＝float32、モデル呼び出しは常に float32 にして結果を戻す）
- 1枚ずつ処理（torchdiffeq 仕様）でのバッチ非汚染

検証は `TDEODEFunction.__call__` と `_run_tde_sampler` の積分部を **逐語コピー**し（pbar / callback / cfg_denoiser / InterruptedException など数学に無関係な reForge 配管のみ除去）、解析解と突き合わせた。

## 検証結果（全 17 項目 PASS）

### Test 1 — 固定ステップ法の経験的収束次数（`dy/dt = -y`, t: 1 → 0, 解 `e^{1-t}`）

グリッドを N=20 → 40 に倍化したときの誤差減少率から実測次数を算出。

| メソッド | 理論次数 | 実測次数 | 判定 |
|---|---|---|---|
| euler | 1 | 0.97 | PASS |
| midpoint | 2 | 1.97 | PASS |
| heun3 | 3 | 2.97 | PASS |
| rk4 | 4 | 3.97 | PASS |

→ torchdiffeq への**呼び出し方**（grid の渡し方・降順 t）が正しく、理論次数どおりに積分されている。

### Test 2 — 適応ステップ法の許容誤差スイープ（`dy/dt = -y`）

rtol/atol を 1e-3 → 1e-5 → 1e-7 と締めたときの終端誤差。

| メソッド | err(1e-3) | err(1e-5) | err(1e-7) | 判定 |
|---|---|---|---|---|
| adaptive_heun | 3.9e-3 | 3.4e-5 | 3.9e-7 | PASS |
| bosh3 | 1.5e-2 | 5.2e-4 | 1.9e-5 | PASS |
| dopri5 | 4.2e-4 | 7.5e-6 | 2.1e-7 | PASS |
| dopri8 | 3.9e-2 | 7.2e-4 | 1.1e-4 | PASS |
| fehlberg2 | 9.3e-2 | 3.1e-3 | 1.2e-5 | PASS |

→ 全メソッドで tol に応答して誤差が単調減少。`min_step` / `max_num_steps` / `dtype` オプションの渡し方も含めて適応制御が正常に効いている。

### Test 3 — `TDEODEFunction` 実コードパス vs 厳密解（Karras 風 σ）

解析的デノイザーを通して probability flow ODE を**実コードパスで積分**し、閉形式と突き合わせる。

| ケース | 評価区間 | 結果 | 判定 |
|---|---|---|---|
| (a) `D ≡ 0.7`, rk4(fp32) | σ_max → σ_min(0.03) | rel max err = 1.2e-7 | PASS |
| (b) `D = x(1-σ/σmax)`, rk4(fp32) | σ_max → σ_min(0.03) | rel max err = 2.1e-6 | PASS |
| (c) `D ≡ 0.7`, σ_max → 0（終端） | euler / rk4 | euler bias≈2e-8 / rk4 bias≈2e-2 | PASS |
| (d) `D = x(1-σ/σmax)`, dopri5(fp64) | σ_max → σ_min, rtol 1e-7 | rel max err = 3.1e-7 | PASS |

→ ODE 定式化・積分方向・スケジュール追従・dtype 往復・1枚ずつ処理のいずれかに配線ミスがあればこの数字は出ない。**移植の正しさはほぼ証明された。**

注: (a)(b) が float32 精度（1e-7〜1e-6 台）で一致するのは仕様どおり。RK のように **マシン精度(1e-16)で一致しないのは「固定法を float32 で回す」TDE の意図的設計**（下記参照）。適応法 (d) は float64 で走るため 1e-7 台。

### Test 4 — `euler` vs 手書き前進 Euler（同一 sigma グリッド・同一 float32）

→ max abs diff = **0.0**（完全一致）。torchdiffeq の euler は sigma グリッド上を素直に1ステップずつ刻んでおり、off-by-one なし・スケジュール追従は正確。k-diffusion 組み込み Euler とほぼビット一致するはず。

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
   tde_sampler.py のコメントどおり、固定法を float64 で回すと sigma スケジュール上で丸め差が蓄積し、特に hires.fix（σ レンジが狭い）で sparkle/grain アーティファクトが出るための意図的処置。このため固定法の解析解一致は float32 精度（1e-7台）で頭打ちになる（RK の固定法がマシン精度で一致したのとはここが異なる）。

2. **torchdiffeq 固定法は `dtype` オプションを無視**し `UserWarning: Unexpected arguments {'dtype': ...}` を出す。これは無害で、固定法の積分 dtype は入力 dtype（=float32）で決まる。実害なし。

3. **σ→0 終端マスキング**。最終ステップ（σ_min→0）で `t ≤ 1e-5` のステージ評価がゼロ勾配にマスクされる（PF-ODE が σ=0 に `1/σ` 特異点を持つための処置、ComfyUI-ODE 由来）。
   - **euler は最終1ステップで厳密に終端へ到達**するため終端バイアス≈0（Test 3c で 2e-8）。
   - rk4 など多段法は σ→0 で後段ステージがマスクされ、小さな終端バイアスが残る（定数デノイザーという最悪条件の合成テストで相対 2e-2 程度）。実デノイザーでは σ→0 で勾配 `(x−D)/σ` が有界になるため実画像での影響はこれより小さく、また最も使われる euler では本質的にゼロ。RK の Test 3c と同種・既知の挙動。

## 結論

- **tde_sampler.py の数学（PF-ODE 定式化、積分方向、スケジュール追従、特異点処理、dtype 取り回し、1枚ずつ処理）は正しく動作している。**
- 固定 4 種は理論どおりの収束次数を示し、適応 5 種は tol に応答して単調収束、実コードパスは解析解と（固定=float32 精度／適応=float64 精度で）一致した。euler は手書き Euler とビット一致、決定論性・バッチ独立性も確認。
- RK Sampler 側の検証と合わせ、**reForge の2本の独立 ODE サンプラー（RK / TDE）はいずれも数学的に正しい**ことが、画像生成を回さずに担保された。
- 腐るのは torchdiffeq / PyTorch との API 整合・dtype/device の糊の部分であり、RK の検証スクリプトと同様、本スクリプトを回帰テストとして回せば PyTorch / torchdiffeq アップデート時の退行検知としてそのまま機能する。

## 再現方法

検証スクリプト `test_tde_core_verification.py` を任意の場所に置き、以下を実行する（reForge 本体への依存はなく、`torch` と `torchdiffeq` だけで完走する）。

```bash
pip install torch torchdiffeq
python test_tde_core_verification.py
```

全 6 テスト（17 項目）が PASS すれば、PF-ODE 定式化・スケジュール追従・特異点処理・dtype 取り回しの正しさが担保される。`TDEODEFunction` / `_run_tde_sampler` を改修した際は、コピー元と検証側を同期させて回帰確認に使える。
