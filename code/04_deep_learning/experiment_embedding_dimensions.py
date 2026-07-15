"""Experiment zum Einfluss verschiedener Embedding-Dimensionen."""

from __future__ import annotations

import csv
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
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


EMBEDDING_DIMENSIONS = [8, 16, 32, 64]


def set_random_seeds() -> None:
    """Setzt Zufallswerte für reproduzierbarere Ergebnisse."""

    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


def build_model(
    embedding_dimension: int,
) -> tf.keras.Model:
    """Erstellt ein Modell mit der angegebenen Embedding-Dimension."""

    model = Sequential(
        [
            Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=embedding_dimension,
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
        name=(
            f"embedding_model_"
            f"{embedding_dimension}_dimensions"
        ),
    )

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def evaluate_model(
    true_labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    """Berechnet die Klassifikationsmetriken."""

    predicted_labels = (
        probabilities >= 0.5
    ).astype(np.int32)

    return {
        "accuracy": float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
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


def run_experiment(
    embedding_dimension: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Trainiert und bewertet eine Embedding-Konfiguration."""

    print(
        "\nExperiment mit "
        f"{embedding_dimension} Embedding-Dimensionen"
    )
    print("-" * 50)

    set_random_seeds()

    model = build_model(
        embedding_dimension=embedding_dimension
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    training_start = time.perf_counter()

    history = model.fit(
        x_train,
        y_train,
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

    probabilities = model.predict(
        x_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    ).reshape(-1)

    metrics = evaluate_model(
        true_labels=y_test,
        probabilities=probabilities,
    )

    result = {
        "embedding_dimension": embedding_dimension,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "training_time_seconds": float(training_time),
        "completed_epochs": int(
            len(history.history["loss"])
        ),
        "parameter_count": int(
            model.count_params()
        ),
        "best_validation_loss": float(
            min(history.history["val_loss"])
        ),
        "best_validation_accuracy": float(
            max(history.history["val_accuracy"])
        ),
    }

    print(
        f"Accuracy: {result['accuracy']:.4f} | "
        f"F1: {result['f1_score']:.4f} | "
        f"Zeit: {result['training_time_seconds']:.2f} s | "
        f"Parameter: {result['parameter_count']}"
    )

    tf.keras.backend.clear_session()

    return result


def save_results(
    results: list[dict[str, float]],
) -> None:
    """Speichert alle Versuchsergebnisse als CSV-Datei."""

    output_path = (
        OUTPUT_DIR
        / "embedding_dimension_results.csv"
    )

    fieldnames = [
        "embedding_dimension",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time_seconds",
        "completed_epochs",
        "parameter_count",
        "best_validation_loss",
        "best_validation_accuracy",
    ]

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nErgebnisse gespeichert: {output_path}")


def create_accuracy_plot(
    results: list[dict[str, float]],
) -> None:
    """Visualisiert Accuracy und F1-Score."""

    dimensions = [
        result["embedding_dimension"]
        for result in results
    ]

    accuracies = [
        result["accuracy"]
        for result in results
    ]

    f1_scores = [
        result["f1_score"]
        for result in results
    ]

    plt.figure(figsize=(9, 6))

    plt.plot(
        dimensions,
        accuracies,
        marker="o",
        label="Accuracy",
    )

    plt.plot(
        dimensions,
        f1_scores,
        marker="o",
        label="F1-Score",
    )

    plt.title(
        "Einfluss der Embedding-Dimension "
        "auf die Modellleistung"
    )
    plt.xlabel("Embedding-Dimension")
    plt.ylabel("Metrikwert")
    plt.xticks(dimensions)
    plt.legend()
    plt.tight_layout()

    result_path = (
        RESULTS_DIR
        / "embedding_dimension_comparison.png"
    )

    plt.savefig(
        result_path,
        dpi=300,
    )
    plt.close()

    print(f"Leistungsvergleich gespeichert: {result_path}")


def create_parameter_plot(
    results: list[dict[str, float]],
) -> None:
    """Visualisiert den Anstieg der Parameterzahl."""

    dimensions = [
        result["embedding_dimension"]
        for result in results
    ]

    parameter_counts = [
        result["parameter_count"]
        for result in results
    ]

    plt.figure(figsize=(9, 6))

    plt.bar(
        dimensions,
        parameter_counts,
        width=5,
    )

    plt.title(
        "Parameterzahl bei verschiedenen "
        "Embedding-Dimensionen"
    )
    plt.xlabel("Embedding-Dimension")
    plt.ylabel("Anzahl trainierbarer Parameter")
    plt.xticks(dimensions)
    plt.tight_layout()

    result_path = (
        RESULTS_DIR
        / "embedding_dimension_parameters.png"
    )

    plt.savefig(
        result_path,
        dpi=300,
    )
    plt.close()

    print(f"Parametervergleich gespeichert: {result_path}")


def main() -> None:
    """Führt alle Embedding-Dimensions-Experimente aus."""

    create_output_directories()

    print("IMDB-Datensatz wird geladen ...")
    dataset = load_imdb_dataset()

    print("Sequenzen werden vorbereitet ...")
    x_train_padded, x_test_padded = (
        prepare_padded_sequences(dataset)
    )

    experiment_results: list[dict[str, float]] = []

    for embedding_dimension in EMBEDDING_DIMENSIONS:
        result = run_experiment(
            embedding_dimension=embedding_dimension,
            x_train=x_train_padded,
            y_train=dataset.y_train,
            x_test=x_test_padded,
            y_test=dataset.y_test,
        )

        experiment_results.append(result)

    save_results(experiment_results)
    create_accuracy_plot(experiment_results)
    create_parameter_plot(experiment_results)

    print("\nZusammenfassung")
    print("---------------")

    for result in experiment_results:
        print(
            f"{result['embedding_dimension']:>2} Dimensionen | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"F1: {result['f1_score']:.4f} | "
            f"Parameter: {result['parameter_count']}"
        )


if __name__ == "__main__":
    main()