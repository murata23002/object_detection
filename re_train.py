# Useful imports
import argparse
import os
import xml.etree.ElementTree as ET

import tensorflow as tf

# Import the same libs that TFLiteModelMaker interally uses
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import (
    train,
    train_lib,
)
from tflite_model_maker import model_spec, object_detector
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker.object_detector import DataLoader


def create_model(train_data, val_data, batch_size, epochs, dotrain):
    try:
        # Create whichever object detector's spec you want
        spec = object_detector.EfficientDetLite1Spec(
            model_name="efficientdet-lite1",
            uri="https://tfhub.dev/tensorflow/efficientdet/lite1/feature-vector/1",
            hparams={"max_instances_per_image": 600},
            model_dir="checkout",
            epochs=epochs,
            batch_size=batch_size,
            steps_per_execution=None,
            moving_average_decay=0,
            # var_freeze_expr="(efficientnet|fpn_cells|resample_p6)",
            var_freeze_expr="fpn_cells|resample_p6)",
            tflite_max_detections=30,
            strategy=None,
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

        return detector, spec
    except Exception as e:
        error_message = f"An error occurred while creating the model: {str(e)}"
        # send_notification(error_message)
        raise e


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

args = parser.parse_args()

# Setup variables
batch_size = 6  # or whatever batch size you want
epochs = 250
checkpoint_dir = "checkout_bk/"  # whatever your checkpoint directory is

# Load you datasets
train_data = load_data(args.train, "train")
test_data = load_data(args.train, "test")
val_data = load_data(args.train, "val")

detector, spec = create_model(train_data, val_data, batch_size, epochs, False)

try:
    # Option A:
    # load the weights from the last successfully completed epoch
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    # Option B:
    # load the weights from a specific checkpoint.
    # Note that there's no ".index" at the end of specific_checkpoint_dir
    # latest = specific_checkpoint_dir

    completed_epochs = int(
        latest.split("/")[-1].split("-")[1]
    )  # the epoch the training was at when the training was last interrupted

    print("Checkpoint found {}".format(latest))
except Exception as e:
    print("Checkpoint not found: ", e)


def export_and_evaluate_model(detector, tfile_name, test_data):
    try:
        export_dir = os.path.join(os.getcwd(), "exported_model")
        os.makedirs(export_dir, exist_ok=True)
        detector.export(export_dir=export_dir, tflite_filename=tfile_name)
        accuracy = detector.evaluate_tflite(
            os.path.join(export_dir, tfile_name), test_data
        )
        print("TFLite model accuracy: ", accuracy)
    except Exception as e:
        error_message = (
            f"An error occurred during model export and evaluation: {str(e)}"
        )
        raise e


"""
Save/export the trained model
Tip: for integer quantization you simply have to NOT SPECIFY 
the quantization_config parameter of the detector.export method.
In this case it would be: 
detector.export(export_dir = export_dir, tflite_filename='model.tflite')
"""
model = spec.create_model()
latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
detector.model = model
export_and_evaluate_model(detector, "model.tflite", test_data)
