"""Training und Bewertung eines neuronalen Sentimentmodells mit Embeddings."""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
)

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    EPOCHS,
    HIDDEN_UNITS,
    MAX_SEQUENCE_LENGTH,
    OUTPUT_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    VOCAB_SIZE,
    create_output_directories,
)
from data_utils import load_imdb_dataset
from prepare_sequences import prepare_padded_sequences


def set_random_seeds() -> None:
    """Setzt Zufallswerte für eine möglichst reproduzierbare Ausführung."""

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def build_embedding_model() -> tf.keras.Model:
    """
    Erstellt ein kleines neuronales Netz für binäre Textklassifikation.

    Die Embedding-Schicht lernt für jede Token-ID einen dichten Vektor.
    """

    model = Sequential(
        [
            Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                input_length=MAX_SEQUENCE_LENGTH,
                mask_zero=True,
                name="word_embedding",
            ),
            GlobalAveragePooling1D(
                name="average_pooling"
            ),
            Dense(
                HIDDEN_UNITS,
                activation="relu",
                name="hidden_layer",
            ),
            Dropout(
                DROPOUT_RATE,
                name="dropout",
            ),
            Dense(
                1,
                activation="sigmoid",
                name="sentiment_output",
            ),
        ],
        name="imdb_embedding_model",
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Berechnet Klassifikationsmetriken."""

    return {
        "accuracy": float(
            accuracy_score(true_labels, predicted_labels)
        ),
        "precision": float(
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
    }


def create_training_history_plot(
    history: tf.keras.callbacks.History,
) -> None:
    """Speichert Loss- und Accuracy-Verläufe."""

    epochs = range(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        history.history["loss"],
        label="Training",
    )
    plt.plot(
        epochs,
        history.history["val_loss"],
        label="Validierung",
    )
    plt.title("Fehlerverlauf des Embedding-Modells")
    plt.xlabel("Epoche")
    plt.ylabel("Binary Cross-Entropy")
    plt.legend()
    plt.tight_layout()

    loss_path = RESULTS_DIR / "embedding_loss_history.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.plot(
        epochs,
        history.history["accuracy"],
        label="Training",
    )
    plt.plot(
        epochs,
        history.history["val_accuracy"],
        label="Validierung",
    )
    plt.title("Genauigkeitsverlauf des Embedding-Modells")
    plt.xlabel("Epoche")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    accuracy_path = (
        RESULTS_DIR / "embedding_accuracy_history.png"
    )
    plt.savefig(accuracy_path, dpi=300)
    plt.close()

    print(f"Loss-Diagramm gespeichert: {loss_path}")
    print(f"Accuracy-Diagramm gespeichert: {accuracy_path}")


def create_confusion_matrix_plot(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    """Speichert die Konfusionsmatrix des neuronalen Modells."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["negativ", "positiv"],
    )

    display.plot(values_format="d")
    plt.title("Konfusionsmatrix: Neuronales Embedding-Modell")
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Tatsächliche Klasse")
    plt.tight_layout()

    result_path = (
        RESULTS_DIR / "confusion_matrix_embedding.png"
    )

    plt.savefig(result_path, dpi=300)
    plt.close()

    print(f"Konfusionsmatrix gespeichert: {result_path}")


def save_metrics(metrics: dict[str, float]) -> None:
    """Speichert die Resultate als JSON."""

    output_path = OUTPUT_DIR / "metrics_embedding.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            metrics,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Metriken gespeichert: {output_path}")


def print_metrics(metrics: dict[str, float]) -> None:
    """Gibt die Resultate im Terminal aus."""

    print("\nErgebnisse des neuronalen Embedding-Modells")
    print("-------------------------------------------")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1_score']:.4f}")
    print(
        f"Trainingszeit:  "
        f"{metrics['training_time_seconds']:.2f} Sekunden"
    )
    print(
        f"Trainierte Epochen: "
        f"{int(metrics['completed_epochs'])}"
    )
    print(
        f"Parameterzahl:  "
        f"{int(metrics['parameter_count'])}"
    )


def main() -> None:
    """Trainiert und bewertet das neuronale Modell."""

    create_output_directories()
    set_random_seeds()

    print("IMDB-Datensatz wird geladen ...")
    dataset = load_imdb_dataset()

    print("Sequenzen werden vereinheitlicht ...")
    x_train_padded, x_test_padded = (
        prepare_padded_sequences(dataset)
    )

    print("Neuronales Modell wird erstellt ...")
    model = build_embedding_model()

    # Erzeugt die Gewichte und zeigt die Architektur.
    model.build(
        input_shape=(None, MAX_SEQUENCE_LENGTH)
    )
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    print("Training wird gestartet ...")
    training_start = time.perf_counter()

    history = model.fit(
        x_train_padded,
        dataset.y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1,
        shuffle=True,
    )

    training_time = (
        time.perf_counter() - training_start
    )

    print("Testdaten werden ausgewertet ...")

    positive_probabilities = model.predict(
        x_test_padded,
        batch_size=BATCH_SIZE,
        verbose=1,
    ).reshape(-1)

    predicted_labels = (
        positive_probabilities >= 0.5
    ).astype(np.int32)

    metrics = calculate_metrics(
        true_labels=dataset.y_test,
        predicted_labels=predicted_labels,
    )

    best_validation_loss = min(
        history.history["val_loss"]
    )

    best_validation_accuracy = max(
        history.history["val_accuracy"]
    )

    metrics.update(
        {
            "training_time_seconds": float(training_time),
            "completed_epochs": int(
                len(history.history["loss"])
            ),
            "parameter_count": int(model.count_params()),
            "best_validation_loss": float(
                best_validation_loss
            ),
            "best_validation_accuracy": float(
                best_validation_accuracy
            ),
            "embedding_dimension": EMBEDDING_DIM,
            "batch_size": BATCH_SIZE,
            "maximum_sequence_length": (
                MAX_SEQUENCE_LENGTH
            ),
            "dropout_rate": DROPOUT_RATE,
            "random_seed": RANDOM_SEED,
        }
    )

    print_metrics(metrics)
    save_metrics(metrics)

    predictions_path = (
        OUTPUT_DIR / "embedding_predictions.npz"
    )

    np.savez_compressed(
        predictions_path,
        true_labels=dataset.y_test,
        predicted_labels=predicted_labels,
        positive_probabilities=positive_probabilities,
    )

    model_path = OUTPUT_DIR / "embedding_model.keras"
    model.save(model_path)

    print(f"Vorhersagen gespeichert: {predictions_path}")
    print(f"Modell gespeichert: {model_path}")

    create_training_history_plot(history)

    create_confusion_matrix_plot(
        true_labels=dataset.y_test,
        predicted_labels=predicted_labels,
    )


if __name__ == "__main__":
    main()