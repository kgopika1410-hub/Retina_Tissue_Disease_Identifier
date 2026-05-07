import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

CLASS_NAMES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferative_DR"]
NUM_CLASSES = 5


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def crop_black_borders(image: np.ndarray, threshold: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > threshold
    if not np.any(mask):
        return image
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return image[y0:y1, x0:x1]


def enhance_image(image: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def preprocess_for_backbone(image: np.ndarray, backbone: str) -> np.ndarray:
    image_255 = image * 255.0
    if backbone == "resnet50":
        image_255 = tf.keras.applications.resnet50.preprocess_input(image_255)
    elif backbone == "vgg16":
        image_255 = tf.keras.applications.vgg16.preprocess_input(image_255)
    else:
        image_255 = tf.keras.applications.efficientnet.preprocess_input(image_255)
    return image_255.astype(np.float32)


def apply_augmentations(image: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        image = np.fliplr(image)
    if np.random.rand() < 0.3:
        angle = np.random.uniform(-12, 12)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if np.random.rand() < 0.3:
        factor = np.random.uniform(0.9, 1.1)
        image = np.clip(image * factor, 0, 1)
    return image


class RetinaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataframe: pd.DataFrame, image_dir: Path, batch_size: int, image_size: tuple[int, int], backbone: str,
                 num_classes: int, shuffle: bool = True, augment: bool = False, apply_enhancement: bool = False,
                 crop_borders: bool = True) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.backbone = backbone
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.augment = augment
        self.apply_enhancement = apply_enhancement
        self.crop_borders = crop_borders
        self.indices = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.dataframe.iloc[batch_indices]

        x_batch = np.zeros((len(batch_df), self.image_size[0], self.image_size[1], 3), dtype=np.float32)
        y_batch = np.zeros((len(batch_df), self.num_classes), dtype=np.float32)

        for i, row in enumerate(batch_df.itertuples(index=False)):
            image_path = self.image_dir / f"{row.id_code}.png"
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_path}")

            if self.crop_borders:
                image = crop_black_borders(image)
            if self.apply_enhancement:
                image = enhance_image(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
            image = image.astype(np.float32) / 255.0

            if self.augment:
                image = apply_augmentations(image)

            x_batch[i] = preprocess_for_backbone(image, self.backbone)
            y_batch[i] = tf.keras.utils.to_categorical(row.diagnosis, num_classes=self.num_classes)

        return x_batch, y_batch

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)


def build_model(backbone: str, input_shape: tuple[int, int, int], num_classes: int,
                backbone_trainable: bool = False) -> tuple[tf.keras.Model, tf.keras.Model]:
    if backbone == "resnet50":
        base_model = tf.keras.applications.ResNet50(include_top=False, weights="imagenet", input_shape=input_shape)
    elif backbone == "vgg16":
        base_model = tf.keras.applications.VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    else:
        base_model = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=input_shape)

    base_model.trainable = backbone_trainable

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float, label_smoothing: float) -> None:
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=["accuracy"],
    )


def unfreeze_top_layers(base_model: tf.keras.Model, num_layers: int) -> None:
    base_model.trainable = True
    if num_layers <= 0:
        return

    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    for layer in base_model.layers[-num_layers:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True


def merge_histories(histories: list[tf.keras.callbacks.History]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            if key not in merged:
                merged[key] = []
            merged[key].extend(values)
    return merged


def plot_training_history(history_data: dict[str, list[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_data.get("accuracy", []), label="Train Accuracy")
    plt.plot(history_data.get("val_accuracy", []), label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history_data.get("loss", []), label="Train Loss")
    plt.plot(history_data.get("val_loss", []), label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close()


def evaluate_model(model: tf.keras.Model, val_generator: RetinaDataGenerator, val_dataframe: pd.DataFrame,
                   output_dir: Path) -> dict:
    probabilities = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = val_dataframe["diagnosis"].to_numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    report_text = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    metrics = {
        "accuracy": float(accuracy),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "f1_weighted": float(f1),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a diabetic retinopathy classifier on APTOS 2019")
    parser.add_argument("--dataset-dir", type=str, default="APTOS 2019 dataset")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--backbone", type=str, default="efficientnetb0", choices=["efficientnetb0", "resnet50", "vgg16"])
    parser.add_argument("--enhance", action="store_true", help="Enable denoising + CLAHE enhancement")
    parser.add_argument("--no-crop", action="store_true", help="Disable black-border cropping")
    parser.add_argument("--augment", action="store_true", help="Enable random augmentation for training")
    parser.add_argument("--finetune", action="store_true", help="Unfreeze pretrained backbone in a second stage")
    parser.add_argument("--finetune-unfreeze-layers", type=int, default=50)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class-weight balancing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    train_csv = dataset_dir / "train.csv"
    train_images_dir = dataset_dir / "train_images"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing file: {train_csv}")
    if not train_images_dir.exists():
        raise FileNotFoundError(f"Missing folder: {train_images_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(train_csv)
    required_columns = {"id_code", "diagnosis"}
    if not required_columns.issubset(set(dataframe.columns)):
        raise ValueError("train.csv must contain id_code and diagnosis columns")

    train_df, val_df = train_test_split(
        dataframe,
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
        stratify=dataframe["diagnosis"],
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(output_dir / "train_split.csv", index=False)
    val_df.to_csv(output_dir / "val_split.csv", index=False)

    image_size = (args.image_size, args.image_size)
    train_generator = RetinaDataGenerator(
        dataframe=train_df,
        image_dir=train_images_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        shuffle=True,
        augment=args.augment,
        apply_enhancement=args.enhance,
        crop_borders=not args.no_crop,
    )

    val_generator = RetinaDataGenerator(
        dataframe=val_df,
        image_dir=train_images_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        backbone=args.backbone,
        num_classes=NUM_CLASSES,
        shuffle=False,
        augment=False,
        apply_enhancement=args.enhance,
        crop_borders=not args.no_crop,
    )

    model, base_model = build_model(
        backbone=args.backbone,
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=NUM_CLASSES,
        backbone_trainable=False,
    )
    compile_model(model, learning_rate=1e-4, label_smoothing=args.label_smoothing)

    class_weight = None
    if not args.no_class_weights:
        class_labels = train_df["diagnosis"].to_numpy()
        classes = np.arange(NUM_CLASSES)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=class_labels)
        class_weight = {int(cls): float(weight) for cls, weight in zip(classes, weights)}
        print(f"Using class weights: {class_weight}")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(output_dir / "best_model.keras"), monitor="val_accuracy", save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    history_stage1 = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    histories = [history_stage1]

    if args.finetune or args.finetune_epochs > 0:
        unfreeze_top_layers(base_model, args.finetune_unfreeze_layers)
        compile_model(model, learning_rate=2e-5, label_smoothing=args.label_smoothing)
        history_stage2 = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=args.finetune_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
        histories.append(history_stage2)

    plot_training_history(merge_histories(histories), output_dir)

    best_model_path = output_dir / "best_model.keras"
    if best_model_path.exists():
        model = tf.keras.models.load_model(best_model_path)

    metrics = evaluate_model(model, val_generator, val_df, output_dir)
    model.save(output_dir / "final_model.keras")

    print("Training completed.")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation Precision (weighted): {metrics['precision_weighted']:.4f}")
    print(f"Validation Recall (weighted): {metrics['recall_weighted']:.4f}")
    print(f"Validation F1-score (weighted): {metrics['f1_weighted']:.4f}")
    print(f"Artifacts saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
