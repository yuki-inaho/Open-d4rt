# LLMオンボーディングサマリー

> このドキュメントは、新任LLMエージェントがOpenD4RTリポジトリで作業を始める際の初期資料です。
> まず一次資料を読み、ローカル環境とGPU/CUDA条件を確認してから作業してください。

## 1. プロジェクト概要と目的
- **プロジェクト名称・領域:** OpenD4RT。D4RT論文の非公式PyTorch/GPU実装で、動画から4D再構成、疎な点追跡、点群可視化、WorldTrack評価を行う。
- **最終成果物:** 学習・評価・推論・Viser可視化が再現可能なコード、checkpoint、設定、データ準備手順、デモ生成フロー。
- **ビジネス背景・価値:** 動画ベースの動的シーン理解、3D/4D再構成、追跡評価の実験基盤として使う。カスタム動画や研究データで素早く動作確認できることが重要。
- **現時点の進捗サマリ:** WorldTrack評価、学習コード、Viser demo、Hugging Face checkpoint導線がある。uv環境とローカル動画/GIFからの軽量demo生成スクリプトを追加済み。さらに推論結果の可視化サブシステム(Rerun ライブラリ/CLI + Gradio アプリ)と、D4RT 推論軌跡 vs COLMAP の一致性チェッカーを追加済み(`docs/visualization_pipeline.md` 参照)。

## 2. クリティカルな要求・制約
> 「壊してはいけない」品質・仕様ラインを箇条書きで列挙します。
- READMEの標準WorldTrack評価フローと既存スクリプトのCLI互換性を壊さない。
- checkpointとmodel configの対応を必ず検証する。特に32CLIP/48CLIPの取り違えは避ける。
- Blackwell GPUでは古い`torch==2.6.0+cu124`が`sm_120`非対応のため、CUDA実演算まで確認する。
- `/checkpoints/**/*.ckpt`、`/checkpoints/**/*.pth`、`/data`、`/tmp`、`.venv/`などの重い生成物やローカル環境はcommitしない。
- WorldTrack `.npz` がない環境では評価結果を主張しない。ローカル動画demoは動作確認用であり、定量評価とは分ける。
- 屋外自動運転系や暗所、48 clip超えの長尺追跡は既知の弱点があるため、精度保証のように断定しない。

## 3. 参照すべき合意済み資料
> 新任エージェントが必ず確認すべき一次資料の一覧です。パスと役割を記載します。

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 要求定義書 | `README.md` | プロジェクト概要、インストール、checkpoint、WorldTrack、評価、demo手順の入口。 |
| 要件定義書 | `docs/data_schema.md`, `configs/model_effective.yaml`, `configs/train_effective.yaml` | データ形式、モデル設定、学習設定の一次情報。 |
| WBS / 進捗 | `README.md`のNews/ToDo, `docs/training.md` | 公開済み機能、学習手順、必要checkpoint、dataset rootの整理。 |
| テスト資産 | `run_eval_worldtrack.sh`, `run_build_worldtrack_demo.sh`, `scripts/build_demo_from_video.py` | 評価、WorldTrack demo、ローカル動画demoの動作確認コマンド。 |
| 既知課題リスト | upstream issues: <https://github.com/Lijiaxin0111/Open-d4rt/issues> | custom dataset、KITTI、点群品質、demo可視化に関する既知論点。 |
| データセット資料 | `docs/dataset/README.md`, `docs/dataset/*.md` | 学習データセットの配置、構造、注意点。 |
| 可視化サブシステム | `docs/visualization_pipeline.md`, `vis/rerun_visualize.py`, `scripts/visualize_rerun.py`, `scripts/demo_gradio.py`, `scripts/dump_static_tracks_for_trajectory.py`, `scripts/check_colmap_trajectory_consistency.py` | demo package / static-tracks / COLMAP軌跡一致性 の Rerun・Gradio 可視化と設計概要。 |

## 4. タスク境界（任せること / 任せないこと）
### 任せるタスク（例）
- uv環境、依存関係、CUDA/PyTorch実演算の確認。
- demo package生成、WorldTrack評価スクリプト、ローカル動画/GIF推論の改善。
- 小さく検証可能なバグ修正、CLIの堅牢化、README/docsの更新。
- upstream issueから再現手順や改善案を抽出し、既存コードの範囲で反映する。

### 任せないタスク（例）
- checkpoint、学習データ、評価データなどの大容量ファイルのcommit。
- 現checkpointで未対応のデータ領域について、精度を保証するような記述。
- 既存評価プロトコルの意味を変える変更。必要な場合は別名CLIや明示的オプションで追加する。
- ユーザーの未commit作業や生成物を勝手に削除、reset、revertすること。

## 5. インタラクション方針
- **回答スタイル:** 日本語で簡潔に、結論、根拠、実行コマンド、残課題を分けて書く。
- **回答手順:** 現状確認、変更内容、検証結果、次のアクションの順に報告する。
- **禁止事項・注意:** 未確認の性能や最新情報を断定しない。WorldTrack評価とローカルdemoを混同しない。Git操作は対象ファイルを確認してから行う。
- **秘匿情報の扱い:** GitHub token、SSH鍵、ローカルパスに含まれる個人情報、非公開データセットは出力・commitしない。

## 6. 試行タスク（オンボーディング演習）
> 小さな検証タスクを2〜3件記載してください。理解度を確認するために実施します。

1. `uv sync --extra vis` 後に `uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` を実行し、CUDA実演算が通るか確認する。
2. `scripts/build_demo_from_video.py` で `demo/softball_25_rgb_gt_pred_2d.gif` から `tmp/local_video_demo` を生成し、`manifest.json` と `assets/demo_data.json` が作られることを確認する。
3. `run_build_worldtrack_demo.sh` を読み、WorldTrack `.npz` が必要な箇所、checkpoint path、軽量化用環境変数を説明する。

## 7. 運用ルール・変更管理
- **ドキュメント更新時の記載ルール:** 変更したCLI、依存、データ配置、制約はREADMEまたはdocsに追記する。検証コマンドも併記する。
- **TBDの扱い:** 未確認事項はTBDとして残し、必要な入力、確認コマンド、判断基準を書く。
- **レビュー/承認フロー:** コード変更は `git diff --check`、構文チェック、該当CLIの軽量実行を通してからcommitする。
- **その他の運用ルール:** 生成物は原則 `tmp/` 以下に出す。大容量ファイルは `.gitignore` の方針に従い、symlinkやローカル配置をcommitしない。

## 8. 公式デモデータで推論 → Rerun(.rrd)を回す手順

公式デモGIF(`demo/*_rgb_gt_pred_2d.gif`, 各32フレーム/640×360)を入力に、D4RT 密推論 → RGB・推定深度・点群・カメラfrustum を含む `.rrd` を生成する一連手順。GPUが必要(CUDA実演算)。

```bash
# 0) 環境(可視化 + screenshot 用の依存)
uv sync --extra vis --extra dev
uv run playwright install chromium    # screenshot モードを使う場合のみ

# 1) 公式GIF → dense scene npz(D4RT推論。クエリ点ベースなので密な正則格子で推論)
B=checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG
uv run python scripts/dump_dense_scene_for_rerun.py \
  --config $B/model.yaml --ckpt-path $B/opend4rt.ckpt \
  --video demo/softball_25_rgb_gt_pred_2d.gif \
  --output tmp/official_dense_32.npz \
  --num-frames 32 --grid-cols 256 --grid-rows 144 --query-chunk-size 512 --device cuda

# 2) npz → .rrd(ヘッドレス。+ PNGスクショも撮るなら --mode screenshot)
uv run --extra vis --extra dev python scripts/visualize_rerun.py \
  --dense-scene tmp/official_dense_32.npz \
  --mode screenshot --output tmp/official_scene_32.rrd --screenshot tmp/official_scene_32.png

# 3) 閲覧(ローカルビューア / Webビューア)
uv run rerun tmp/official_scene_32.rrd
uv run rerun --serve-web tmp/official_scene_32.rrd   # 表示URLをブラウザで開く
```

- **表示パネル:** 3Dビュー(色付き点群 + カメラfrustumのワイヤフレーム)、`frame/image`(RGB)、`frame/depth`(推定深度)。
- **カメラ:** D4RTは姿勢を直接出さない(`pred_camera_*=None`)ため、静的点の2D-3D対応からPnPで各フレーム姿勢を復元している。
- **深度:** クエリ点zから格子解像(例 256×144)の低解像近似として生成。毎ピクセル深度ではない。
- **規模/VRAM実測:** 32フレーム・36,864点・`--query-chunk-size 512` で **ピーク約6.0GB / 24GB**(約25%)。VRAMは制約にならず、上限はGIF長(32)とモデル入力解像(256×256)。VRAMを使い切るには `--query-chunk-size` を上げる(速度向上)。
- **2000フレーム源(COLMAP元データ)で回す場合:** `--video` の代わりに `--image-dir recon/.../images --num-frames 24 --frame-stride 3`。D4RTは≤48フレームのクリップ単位なので2000全体の一括処理は不可、代表クリップを処理する。
- 設計の詳細は `docs/visualization_pipeline.md`(特に §22.5)を参照。生成物は `tmp/` 配下(`.gitignore` 済み)。

---

### 付録: 参考情報
- **主要リポジトリ/ディレクトリ:** upstream `Lijiaxin0111/Open-d4rt`、fork `yuki-inaho/Open-d4rt`、実装 `src/`、可視化 `vis/`、設定 `configs/`、資料 `docs/`。
- **代表的なコマンド:**
  - `uv sync --extra vis`
  - `uv run python scripts/build_demo_from_video.py --config checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/model.yaml --ckpt-path checkpoints/OpenD4RT_48CLIP_9Mix_NoCropAUG/opend4rt.ckpt --input demo/softball_25_rgb_gt_pred_2d.gif --output-dir tmp/local_video_demo --num-frames 4 --device cuda`
  - `uv run python vis/serve_demo_viser.py --root tmp/local_video_demo --port 8082`
  - `LIMIT_SEQS=1 SUBSETS=adt_mini OUTPUT_DIR=tmp/eval_smoke bash run_eval_worldtrack.sh`
  - 可視化(Rerun, ヘッドレス): `uv run --extra vis python scripts/visualize_rerun.py --demo-package tmp/local_video_demo --mode rrd --output outputs/demo.rrd`
  - 軌跡一致性 + スクショ: `uv run --extra vis --extra dev python scripts/visualize_rerun.py --tracks-npz <pred.npz> --colmap-model <sparse_txt> --mode screenshot --output outputs/traj.rrd --screenshot outputs/traj.png`
  - Gradio 閲覧: `uv run --extra vis python scripts/demo_gradio.py --results-root tmp/`(既定 127.0.0.1:7860)
- **依存ライブラリ:** PyTorch、TorchVision、NumPy、OpenCV headless、Pillow、Matplotlib、TensorBoard、lz4、Viser。可視化 extra(`--extra vis`)で Rerun (`rerun-sdk`)、Gradio、trimesh を追加。screenshot モードは `--extra dev` の Playwright CLI + chromium(`uv run playwright install chromium`)を使用。Blackwell GPUではCUDA 13.0系PyTorch wheelを優先する。
- **連絡先/責任者:** TBD。GitHub上ではfork所有者 `yuki-inaho`、upstream所有者 `Lijiaxin0111` を確認する。

> ※テンプレートは必要に応じて拡張・縮退して構いません。記入済みのドキュメントはバージョン管理してください。
