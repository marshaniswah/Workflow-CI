from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


def prepare_datasets(dataset_dir: Path, image_size: int, batch_size: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "val",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "test",
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), val_ds.prefetch(autotune), test_ds.prefetch(autotune), class_names


def build_model(image_size: int, num_classes: int, learning_rate: float):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="natural_scene_classifier_ci")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_artifacts(model, test_ds, class_names: list[str], history):
    artifacts_dir = Path("training_artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(artifacts_dir / "training_curve.png", dpi=150)
    plt.close(fig)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probabilities, axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    (artifacts_dir / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(artifacts_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    mlflow.log_artifacts(str(artifacts_dir), artifact_path="training_artifacts")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="dataset_preprocessing")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    mlflow.set_experiment("Natural Scene Classification CI")
    mlflow.tensorflow.autolog(log_models=True)

    dataset_dir = Path(args.dataset_dir)
    train_ds, val_ds, test_ds, class_names = prepare_datasets(dataset_dir, args.image_size, args.batch_size)

    with mlflow.start_run(run_name="github_actions_training") as run:
        Path("run_id.txt").write_text(run.info.run_id, encoding="utf-8")
        mlflow.log_params(
            {
                "image_size": args.image_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "architecture": "MobileNetV2 transfer learning",
                "dataset": "Intel Image Classification",
                "classes": ",".join(class_names),
            }
        )
        model = build_model(args.image_size, len(class_names), args.learning_rate)
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
        test_metrics = model.evaluate(test_ds, return_dict=True)
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", float(metric_value))
        report = save_artifacts(model, test_ds, class_names, history)
        mlflow.log_metric("test_macro_f1", float(report["macro avg"]["f1-score"]))
        model.save("model.keras")
        mlflow.log_artifact("model.keras", artifact_path="keras_model")
        mlflow.tensorflow.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()
