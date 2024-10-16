import argparse
import os
import xml.etree.ElementTree as ET

import requests
from tflite_model_maker import model_spec, object_detector


# Google Chatへの通知を送信する関数
def send_notification(message):
    url = ""
    if url:
        data = {"text": message}
        response = requests.post(url, json=data)
        if response.status_code != 200:
            print("Failed to send notification.")


def read_xml(directory):
    class_list = set()
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            xml_file = os.path.join(directory, filename)

            tree = ET.parse(xml_file)
            root = tree.getroot()

            for obj in root.findall("object"):
                name = obj.find("name").text
                class_list.add(name)

    return list(class_list)


def load_data(train_dir, target):
    dir_path = os.path.join(train_dir, target)
    img_dir = os.path.join(dir_path, "images")
    ann_dir = os.path.join(dir_path, "annotations")
    classes = read_xml(ann_dir)
    print("read directory = ", ann_dir)
    print("class list = ", classes)
    data = object_detector.DataLoader.from_pascal_voc(img_dir, ann_dir, classes)
    return data


def create_detector(
    train_data, val_data, model_name, batch_size, epochs, dotrain, checkout
):
    try:
        # Create whichever object detector's spec you want
        spec = object_detector.EfficientDetLite2Spec(
            model_name="efficientdet-lite2",
            uri="https://tfhub.dev/tensorflow/efficientdet/lite2/feature-vector/1",
            hparams={"max_instances_per_image": 50},
            model_dir=checkout,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_execution=None,
            moving_average_decay=0,
            # var_freeze_expr="(efficientnet|fpn_cells|resample_p6)",
            var_freeze_expr="fpn_cells|resample_p6)",
            tflite_max_detections=50,
            strategy="gpus",  # gpus None
            tpu=None,
            gcp_project=None,
            tpu_zone=None,
            use_xla=False,
            profile=False,
            debug=False,
            tf_random_seed=111111,
            verbose=1,
        )
        # Create the object detector
        detector = object_detector.create(
            train_data,
            model_spec=spec,
            batch_size=batch_size,
            train_whole_model=True,
            validation_data=val_data,
            epochs=epochs,
            do_train=dotrain,
        )

        return detector
    except Exception as e:
        error_message = f"An error occurred while creating the model: {str(e)}"
        send_notification(error_message)
        raise e


def export_and_evaluate_model(model, tfile_name, test_data):
    try:
        # model.exportをカレントディレクトリ内にディレクトリを作成してその中にファイルを出力するように
        export_dir = os.path.join(os.getcwd(), "exported_model")
        os.makedirs(export_dir, exist_ok=True)
        model.export(export_dir=export_dir, tflite_filename=tfile_name)
        accuracy = model.evaluate_tflite(
            os.path.join(export_dir, tfile_name), test_data
        )
        send_notification(
            f"TFLite model accuracy: {accuracy}",
        )
        send_notification("モデルのエクスポートと評価が完了しました。")
    except Exception as e:
        error_message = (
            f"An error occurred during model export and evaluation: {str(e)}"
        )
        send_notification(error_message)
        raise e


def main():
    try:
        parser = argparse.ArgumentParser(
            description="Object Detector Training Arguments",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "--train",
            required=True,
            type=str,
            default="",
            help="Path to train data folder",
        )
        parser.add_argument(
            "--test",
            required=True,
            type=str,
            default="",
            help="Path to test data folder",
        )

        parser.add_argument(
            "--val",
            required=True,
            type=str,
            default="",
            help="Path to validation data folder",
        )

        parser.add_argument(
            "--batch", type=int, default=25, help="Batch size for training"
        )
        parser.add_argument(
            "--epochs", type=int, default=50, help="Number of epochs for training"
        )
        parser.add_argument(
            "--model", type=str, default="efficientdet-lite0", help="Select model name "
        )
        parser.add_argument(
            "--tfilteName",
            type=str,
            default="model.tflite",
            help="Create model tflite name",
        )
        parser.add_argument(
            "--checkout",
            type=str,
            default="checkout",
            help="Create model tflite name",
        )
        parser.add_argument("--dotrain", type=bool, default=True, help="train")
        args = parser.parse_args()

        # 指定コマンドで存在しないディレクトリが指定されたときにエラーを表示
        for dir_path in [args.train, args.test]:
            if not os.path.exists(dir_path):
                error_message = f"Error: Directory '{dir_path}' does not exist."
                send_notification(error_message)
                return

        train_data = load_data(args.train, "train")
        test_data = load_data(args.test, "test")
        val_data = load_data(args.test, "val")
        print(train_data)
        print(test_data)
        print(val_data)

        model = create_detector(
            train_data,
            val_data,
            args.model,
            args.batch,
            args.epochs,
            args.dotrain,
            args.checkout,
        )
        export_and_evaluate_model(model, args.tfilteName, test_data)
        send_notification("created model")
    except Exception as e:
        error_message = f"An unexpected error occurred: {str(e)}"
        send_notification(error_message)
        raise e


if __name__ == "__main__":
    main()
