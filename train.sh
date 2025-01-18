#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)  # 現在の日付と時刻を取得して、YYYYMMDD_HHMMSS形式で保存

SCRIPT_PATH=$(realpath "$0")  # このスクリプト自体の絶対パスを取得（realpathコマンドを使用）
DATASET_NAME="SampleDataSet"  # データセットの名前を指定
CHECKOUT_DIR="checkout/${DATASET_NAME}_$DATE"  # 出力ファイルを保存するディレクトリ名を作成（日付と名前を組み合わせて一意に）

mkdir -p "$CHECKOUT_DIR"  # 指定したディレクトリを作成（既に存在している場合はエラーなしでスキップ）

# このスクリプトを作成したディレクトリにコピー（再現性を確保するため）
cp "$SCRIPT_PATH" "$CHECKOUT_DIR"

# データセットディレクトリを指定（サブディレクトリはPython側で検索する）
TRAIN_PATH="./datasets/${DATASET_NAME}"  # トレーニングデータセットのディレクトリ
TEST_PATH="./datasets/${DATASET_NAME}"   # テストデータセットのディレクトリ
VAL_PATH="./datasets/${DATASET_NAME}"    # 検証データセットのディレクトリ

# モデルのトレーニングを実行するPythonスクリプトを実行
python train.py \
    --train "$TRAIN_PATH" \           # トレーニングデータセットのパス
    --test "$TEST_PATH" \             # テストデータセットのパス
    --val "$VAL_PATH" \               # 検証データセットのパス
    --batch 6 \                       # バッチサイズ（同時に処理するデータ数）
    --epochs 500 \                    # エポック数（データセット全体を繰り返す回数）
    --tfilteName ${DATASET_NAME}_$DATE.tflite \  # トレーニング後に出力するTensorFlow Liteファイルの名前
    --checkout "$CHECKOUT_DIR" \      # モデルのチェックポイントやログを保存するディレクトリ
    --freeze '(efficientnet|fpn_cells|resample_p6)' \ # トレーニング中に凍結（更新しない）レイヤーの指定
    --doTrain \                       # トレーニングを実行するフラグ（このフラグがないとトレーニングは実行されない）
    # --resumeCheckpoint "./checkout/${DATASET_NAME}" # データセットに対応したチェックポイントのディレクトリ