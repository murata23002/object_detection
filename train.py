import argparse
import os
import xml.etree.ElementTree as ET
import requests
import logging
import tensorflow as tf 
from tflite_model_maker import object_detector
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train
from tensorflow_examples.lite.model_maker.third_party.efficientdet.keras import train_lib

def setup_logging(checkpoint_dir):
    log_file = os.path.join(checkpoint_dir, "training.log")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])

def send_notification(message):
    logging.info(f"Sending notification: {message}")
    url = ""
    if url:
        data = {"text": message}
        response = requests.post(url, json=data)
        if response.status_code != 200:
            logging.error("Failed to send notification.")

def read_xml(directory):
    class_list = set()
    logging.info(f"Reading XML files from directory: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):
            xml_file = os.path.join(directory, filename)
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for obj in root.findall("object"):
                name = obj.find("name").text
                class_list.add(name)
    logging.info(f"Found classes: {class_list}")
    return list(class_list)

def load_data(data_dir, target):
    dir_path = os.path.join(data_dir, target)
    img_dir = os.path.join(dir_path, "images")
    ann_dir = os.path.join(dir_path, "annotations")
    classes = read_xml(ann_dir)
    classes = sorted(classes)
    logging.info(f"Reading directory: {ann_dir}, Class list: {classes}")
    return object_detector.DataLoader.from_pascal_voc(img_dir, ann_dir, classes)

def custom_get_callbacks(params, val_dataset=None):

    callbacks = train_lib.get_callbacks(params, val_dataset)
    callbacks = [cb for cb in callbacks if not isinstance(cb, tf.keras.callbacks.ModelCheckpoint)]
    
    recent_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(params['model_dir'], 'ckpt-{epoch:d}.h5'),
        verbose=params['verbose'],
        save_freq=params['save_freq'],
        save_weights_only=True,
        max_to_keep=6
    )

    best_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(params['model_dir'], 'best_weights.h5'),
        verbose=params['verbose'],
        save_weights_only=True,
        monitor='val_loss',
        save_best_only=True,
        mode='min' 
    )

    callbacks = [recent_ckpt_callback, best_ckpt_callback]

    return callbacks

def model_train(detector, train_data, val_data, batch_size, epochs, resume_checkpoint):
    train_ds, steps_per_epoch, _ = detector._get_dataset_and_steps(train_data, batch_size, is_training=True)
    validation_ds, validation_steps, val_json_file = detector._get_dataset_and_steps(val_data, batch_size, is_training=False)
    model = detector.model
    config = detector.model_spec.config
    config.update(dict(
        steps_per_epoch=steps_per_epoch,
        eval_samples=batch_size * validation_steps,
        val_json_file=val_json_file,
        batch_size=batch_size
    ))

    train.setup_model(model, config)
    train.init_experimental(config)

    if resume_checkpoint:
        logging.info(f"Resuming training from checkpoint: {resume_checkpoint}")
        latest = tf.train.latest_checkpoint(resume_checkpoint)
        if latest:
            completed_epochs = int(latest.split("/")[-1].split("-")[1])
            model.load_weights(latest)
            logging.info(f"Checkpoint found: {latest}")
        else:
            logging.warning(f"No checkpoint found in {resume_checkpoint}. Starting training from scratch.")
            completed_epochs = 0
    else:
        logging.info("No checkpoint provided. Starting training from scratch.")
        completed_epochs = 0

    callbacks = custom_get_callbacks(config.as_dict(), validation_ds)

    model.fit(
        train_ds,
        epochs=epochs,
        initial_epoch=completed_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_ds,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

def create_detector(train_data, val_data, batch_size, epochs, checkout, resume_checkpoint, freeze_layers):
    try:
        var_freeze_expr = freeze_layers if freeze_layers else ""
        logging.info("Creating EfficientDet model...")
        
        spec = object_detector.EfficientDetLite4Spec(
            model_name="efficientdet-lite4",
            uri="https://tfhub.dev/tensorflow/efficientdet/lite4/feature-vector/2",
            hparams={"grad_checkpoint": True, "max_instances_per_image": 2000},
            model_dir=checkout,
            epochs=epochs,
            batch_size=batch_size,
            var_freeze_expr=var_freeze_expr,
            strategy="gpus",
            tflite_max_detections=50,
            tf_random_seed=111111,
            verbose=1,
        )

        with spec.ds_strategy.scope():
            detector = object_detector.create(
                train_data,
                model_spec=spec,
                batch_size=batch_size,
                train_whole_model=False,
                validation_data=val_data,
                epochs=epochs,
                do_train=False,
            )
            model_train(detector, train_data, val_data, batch_size, epochs, resume_checkpoint)

        return detector
    except Exception as e:
        error_message = f"An error occurred while creating the model: {str(e)}"
        logging.error(error_message)
        send_notification(error_message)
        raise e

def export_and_evaluate_model(model, tfile_name, test_data):
    try:
        logging.info("Exporting and evaluating the model...")
        export_dir = os.path.join(os.getcwd(), "exported_model")
        os.makedirs(export_dir, exist_ok=True)
        model.export(export_dir=export_dir, tflite_filename=tfile_name)
        accuracy = model.evaluate_tflite(os.path.join(export_dir, tfile_name), test_data)
        logging.info(f"TFLite model accuracy: {accuracy}")
        send_notification(f"TFLite model accuracy: {accuracy}")
        send_notification("モデルのエクスポートと評価が完了しました。")
    except Exception as e:
        error_message = f"An error occurred during model export and evaluation: {str(e)}"
        logging.error(error_message)
        send_notification(error_message)
        raise e

def main():
    try:
        parser = argparse.ArgumentParser(description="Object Detector Training Arguments")
        parser.add_argument("--train", required=True, type=str, help="Path to train data folder")
        parser.add_argument("--test", required=True, type=str, help="Path to test data folder")
        parser.add_argument("--val", required=True, type=str, help="Path to validation data folder")
        parser.add_argument("--batch", type=int, default=25, help="Batch size for training")
        parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
        parser.add_argument("--tfilteName", type=str, default="model.tflite", help="TFLite model filename")
        parser.add_argument("--checkout", type=str, default="checkout", help="Checkpoint directory")
        parser.add_argument("--resumeCheckpoint", type=str, default="", help="Resume training from checkpoint")
        parser.add_argument("--freeze", type=str, default="", help="Layers to freeze (e.g., 'efficientnet|fpn_cells')")
        args = parser.parse_args()

        setup_logging(args.checkout)

        for dir_path in [args.train, args.test, args.val]:
            if not os.path.exists(dir_path):
                logging.error(f"Error: Directory '{dir_path}' does not exist.")
                send_notification(f"Error: Directory '{dir_path}' does not exist.")
                return

        train_data = load_data(args.train, "train")
        test_data = load_data(args.test, "test")
        val_data = load_data(args.val, "val")

        model = create_detector(
            train_data,
            val_data,
            args.batch,
            args.epochs,
            args.checkout,
            args.resumeCheckpoint,
            args.freeze,
        )

        export_and_evaluate_model(model, args.tfilteName, test_data)
        send_notification("モデルの作成が完了しました。")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        send_notification(f"An unexpected error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
