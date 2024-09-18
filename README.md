
# 概要

この[TensorFlow Lite Model_maker](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/lite/model_maker/README.md)を使用して物体検出モデルをトレーニング、エクスポート、評価するためのものです。EfficientDetモデルを使用しており、Pascal VOCフォーマットのカスタムデータセットでのトレーニングに対応しています。
このTensorFlow Lite Model_makerは現在メンテナンスがあまりされておらずインストールまで大変です。
特定の環境で特定の状態でしか動作しないためDockerを使って実行するのがおすすめです。
また内部で利用されているAPIは、Tensor Flow V1が利用されている模様です。

## 必要条件

### ソフトウェアとライブラリ：
- **Docker**（コンテナ化されたモデルトレーニング用）
- **TensorFlow Model Maker**
- **Python**（バージョン3.6以上）
- **tflite_model_maker**
- **requests**（Google Chatへの通知送信用）
- **argparse**（コマンドライン引数解析用）

### ハードウェア：
- GPU（オプション、トレーニングの高速化に推奨）

### データセット：
- Pascal VOCフォーマットのデータセット（画像とXML形式のアノテーション）

## ディレクトリ構造

```
├── train.py               # メイントレーニングスクリプト
├── Dockerfile              # Docker環境セットアップ用
├── build_docker.sh         # Dockerイメージのビルドスクリプト
├── tensor_flow_lite_image_exec_bash.sh   # Dockerコンテナ実行スクリプト
├── train.sh               # トレーニング開始スクリプト
└── data/
    ├── train//            
    |   ├── images # トレーニング用の画像ディレクトリ
    |   └── annotations # トレーニング用のPascal VOC形式のアノテーションディレクトリ
    ├── val
    |   ├── images # バリデーション用の画像ディレクトリ
    |   └── annotations # バリデーション用のPascal VOC形式のアノテーションディレクトリ
    └── test
        ├── images # テスト用の画像ディレクトリ
        └── annotations # テスト用のPascal VOC形式のアノテーションディレクトリ      /       
```
## セットアップ

### ステップ 1: リポジトリをクローン
```bash
git clone https://github.com/murata23002/object_detection.git
cd object_detection
```

### ステップ 2: データセットの準備
Pascal VOCフォーマットのデータセットを以下のディレクトリに配置します：
```
data/
├── images/
└── annotations/
```

### ステップ 3: Dockerイメージをビルド
```bash
./build_docker.sh
```

### ステップ 4: Dockerコンテナを実行
```bash
./tensor_flow_lite_image_exec_bash.sh
```

### ステップ 5: トレーニングの開始
`train.sh`スクリプト内のデータセットパスを正しく設定し、以下のコマンドを実行します：
```bash
./train.sh
```

## スクリプトの詳細

### `train.py`
- **機能**: EfficientDetモデルをPascal VOCデータを使用してトレーニングし、`.tflite`ファイルにエクスポートして評価します。
- **コマンドライン引数**:
  - `--train`: トレーニングデータフォルダへのパス
  - `--test`: テストデータフォルダへのパス
  - `--val`: 検証データフォルダへのパス
  - `--batch`: トレーニング用のバッチサイズ
  - `--epochs`: トレーニングエポック数
  - `--model`: EfficientDetモデル名（例: efficientdet-lite0）
  - `--tfilteName`: エクスポートするTFLiteファイルの名前
  - `--checkout`: チェックポイント保存用ディレクトリ
  - `--dotrain`: トレーニングを行うかどうかのフラグ

### `build_docker.sh`
- 必要なTensorFlow環境を含むDockerイメージをビルドします。

### `tensor_flow_lite_image_exec_bash.sh`
- GPUサポートを有効にしたDockerコンテナを実行し、プロジェクトを操作可能にします。

### `train.sh`
- 事前に設定したパスやパラメータで`train.py`を実行するスクリプト。TFLiteモデルの名前は実行時の日時で動的に生成されます。

## トレーニング完了通知

トレーニングが完了するとGoogle Chatへ結果の通知を行います。
`train.py`内の以下のGoogle ChatのWebhook URLを、自身のWebhook URLに更新してください。
```python
# Google Chatへの通知を送信する関数
def send_notification(message):
    url = "user chat webhook url" 
    if url:
        data = {"text": message}
        response = requests.post(url, json=data)
        if response.status_code != 200:
            print("Failed to send notification.")

```
## 例

モデルをトレーニングするには、以下のコマンドを実行します：
```bash
python train.py --train ./path/to/dataset \
    --test ./path/to/dataset \
    --val ./path/to/dataset \
    --batch 4 \
    --epochs 500 \
    --tfilteName head_face_body_limb_$DATE.tflite \
    --checkout output/checkout_1_$DATE
```

## エクスポートされたモデル

トレーニングされたモデルは`exported_model/`ディレクトリに保存されます。


