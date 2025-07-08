# PPK ISAM オブジェクト指向実装

RTKLib-pyのPPK ISAMプロセッサーをオブジェクト指向で整理した実装です。

## 構造

### 1. 完全なオブジェクト指向実装 (`isam/` ディレクトリ)

```
isam/
├── core/                # コアコンポーネント
│   ├── config.py        # 設定管理
│   ├── data_loader.py   # データ読み込み
│   ├── optimizer.py     # ISAM2最適化
│   └── ppk_processor.py # メイン処理
├── factors/             # GTSAMファクター
│   ├── gnss_factors.py  # GNSS測定ファクター
│   ├── imu_factors.py   # IMUファクター
│   └── motion_factors.py # 運動制約
└── utils/               # ユーティリティ
```

**使用方法:**
```python
from isam import PPKProcessor, ISAMConfig

config = ISAMConfig(
    datadir='path/to/data',
    rovfile='rover.obs',
    basefile='base.obs',
    navfile='base.nav',
    imufile='imu.csv'
)

processor = PPKProcessor(config)
solutions = processor.process()
```

### 2. シンプルなオブジェクト指向ラッパー (`src/run_ppk_isam_oo.py`)

既存のRTKLib実装をラップしたシンプルなバージョン：

```python
from run_ppk_isam_oo import PPKProcessor, PPKConfig

config = PPKConfig(
    datadir='path/to/data',
    maxepoch=100,
    trace_level=1
)

processor = PPKProcessor(config)
solutions = processor.process()
processor.plot_results()
```

## 特徴

### シンプルラッパー版の利点
- 既存のRTKLib実装を活用
- 最小限の変更で動作
- 安定性が高い
- RTKLibとの互換性維持

### 完全OO版の利点
- クリーンなアーキテクチャ
- 拡張性が高い
- ファクターの追加が簡単
- テストが容易

## 実行例

```bash
# シンプルラッパー版
cd src
uv run python run_ppk_isam_oo.py --maxepoch 100 --trace 1 --plot

# 完全OO版（開発中）
uv run python run_ppk_isam_new.py --maxepoch 100
```

## 今後の改善点

1. 完全OO版のRTKLib依存の解消
2. ファクター実装の充実
3. リアルタイム処理対応
4. より詳細な設定管理