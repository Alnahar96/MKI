"""
Experiment: Einfluss der Trainingsdatengröße auf die Modellleistung.

Für das Experiment wird dieselbe neuronale Netzarchitektur mit
unterschiedlichen Anteilen der verfügbaren Trainingsdaten trainiert.
Dadurch lässt sich untersuchen, wie sich zusätzliche Trainingsbeispiele
auf Genauigkeit, Generalisierungsfähigkeit und Trainingsdauer auswirken.
"""

from __future__ import annotations

import csv
import json
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    EPOCHS,
    MAX_SEQUENCE_LENGTH,
    OUTPUT_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    VOCAB_SIZE,
    create_output_directories,
)


# ---------------------------------------------------------
# Reproduzierbarkeit
# ---------------------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


def load_prepared_sequences() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Lädt die zuvor vorbereiteten und gespeicherten Sequenzen.

    Returns:
        Trainingssequenzen, Trainingslabels,
        Testsequenzen und Testlabels.
    """

    sequences_path = OUTPUT_DIR / "prepared_sequences.npz"

    if not sequences_path.exists():
        raise FileNotFoundError(
            "Die vorbereiteten Sequenzen wurden nicht gefunden:\n"
            f"{sequences_path}\n\n"
            "Bitte zuerst folgenden Befehl ausführen:\n"
            "python prepare_sequences.py"
        )

    with np.load(sequences_path) as data:
        required_keys = {
            "x_train",
            "y_train",
            "x_test",
            "y_test",
        }

        available_keys = set(data.files)
        missing_keys = required_keys - available_keys

        if missing_keys:
            raise KeyError(
                "Im NPZ-Archiv fehlen folgende Datensätze: "
                f"{sorted(missing_keys)}\n"
                f"Vorhandene Datensätze: {sorted(available_keys)}"
            )

        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]

    print("Vorbereitete Daten erfolgreich geladen.")
    print(f"Vollständige Trainingsdaten: {x_train.shape}")
    print(f"Trainingslabels:             {y_train.shape}")
    print(f"Testdaten:                   {x_test.shape}")
    print(f"Testlabels:                  {y_test.shape}")

    return x_train, y_train, x_test, y_test


def create_model() -> tf.keras.Model:
    """
    Erstellt für jeden Versuch ein neues neuronales Netz.

    Die Architektur bleibt in allen Experimenten identisch,
    damit ausschließlich der Einfluss der Trainingsdatengröße
    untersucht wird.
    """

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(MAX_SEQUENCE_LENGTH,),
                name="input_sequence",
            ),

            tf.keras.layers.Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
                mask_zero=True,
                name="word_embedding",
            ),

            tf.keras.layers.GlobalAveragePooling1D(
                name="global_average_pooling",
            ),

            tf.keras.layers.Dense(
                units=64,
                activation="relu",
                name="hidden_dense_layer",
            ),

            tf.keras.layers.Dropout(
                rate=DROPOUT_RATE,
                name="dropout_layer",
            ),

            tf.keras.layers.Dense(
                units=1,
                activation="sigmoid",
                name="output_layer",
            ),
        ],
        name="imdb_embedding_classifier",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_nested_training_indices(
    number_of_available_samples: int,
) -> np.ndarray:
    """
    Erstellt eine feste zufällige Reihenfolge der Trainingsindizes.

    Dadurch sind kleinere Trainingsmengen Teilmengen der größeren.
    Das verbessert die Vergleichbarkeit der Experimente.
    """

    random_generator = np.random.default_rng(RANDOM_SEED)

    return random_generator.permutation(
        number_of_available_samples
    )


def run_training_size_experiment(
    x_train_base: np.ndarray,
    y_train_base: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> list[dict[str, float | int]]:
    """
    Trainiert das Modell mit verschiedenen Trainingsdatenanteilen.
    """

    training_fractions = [
        0.20,
        0.40,
        0.60,
        0.80,
        1.00,
    ]

    fixed_indices = create_nested_training_indices(
        len(x_train_base)
    )

    experiment_results: list[dict[str, float | int]] = []

    for fraction in training_fractions:
        print("\n" + "=" * 65)
        print(
            "Training mit "
            f"{fraction * 100:.0f} % des Trainingspools"
        )
        print("=" * 65)

        number_of_samples = int(
            len(x_train_base) * fraction
        )

        selected_indices = fixed_indices[:number_of_samples]

        x_train_subset = x_train_base[selected_indices]
        y_train_subset = y_train_base[selected_indices]

        print(
            "Anzahl verwendeter Trainingsbeispiele: "
            f"{len(x_train_subset)}"
        )

        tf.keras.backend.clear_session()

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)

        model = create_model()

        start_time = time.perf_counter()

        history = model.fit(
            x_train_subset,
            y_train_subset,
            validation_data=(
                x_validation,
                y_validation,
            ),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=1,
            shuffle=True,
        )

        training_duration = (
            time.perf_counter() - start_time
        )

        validation_loss, final_validation_accuracy = (
            model.evaluate(
                x_validation,
                y_validation,
                verbose=0,
            )
        )

        test_loss, test_accuracy = model.evaluate(
            x_test,
            y_test,
            verbose=0,
        )

        validation_accuracies = history.history[
            "val_accuracy"
        ]

        best_validation_accuracy = float(
            max(validation_accuracies)
        )

        best_epoch = int(
            np.argmax(validation_accuracies) + 1
        )

        final_training_accuracy = float(
            history.history["accuracy"][-1]
        )

        result = {
            "training_fraction": float(fraction),
            "training_percentage": int(fraction * 100),
            "number_of_training_samples": int(
                number_of_samples
            ),
            "final_training_accuracy": (
                final_training_accuracy
            ),
            "final_validation_accuracy": float(
                final_validation_accuracy
            ),
            "best_validation_accuracy": (
                best_validation_accuracy
            ),
            "best_epoch": best_epoch,
            "validation_loss": float(validation_loss),
            "test_accuracy": float(test_accuracy),
            "test_loss": float(test_loss),
            "training_duration_seconds": float(
                training_duration
            ),
        }

        experiment_results.append(result)

        print("\nErgebnis des Versuchs")
        print("---------------------")
        print(
            "Beste Validierungsgenauigkeit: "
            f"{best_validation_accuracy:.4f}"
        )
        print(f"Beste Epoche: {best_epoch}")
        print(
            "Finale Validierungsgenauigkeit: "
            f"{final_validation_accuracy:.4f}"
        )
        print(
            f"Testgenauigkeit: {test_accuracy:.4f}"
        )
        print(
            "Trainingsdauer: "
            f"{training_duration:.2f} Sekunden"
        )

    return experiment_results


def save_results_as_json(
    results: list[dict[str, float | int]],
) -> None:
    """Speichert die Versuchsergebnisse als JSON-Datei."""

    output_path = (
        OUTPUT_DIR / "training_size_experiment_results.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            results,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"JSON-Ergebnisse gespeichert: {output_path}")


def save_results_as_csv(
    results: list[dict[str, float | int]],
) -> None:
    """Speichert die Versuchsergebnisse als CSV-Datei."""

    output_path = (
        OUTPUT_DIR / "training_size_experiment_results.csv"
    )

    if not results:
        raise ValueError(
            "Es sind keine Versuchsergebnisse vorhanden."
        )

    fieldnames = list(results[0].keys())

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

    print(f"CSV-Ergebnisse gespeichert: {output_path}")


def create_accuracy_plot(
    results: list[dict[str, float | int]],
) -> None:
    """
    Erstellt ein Diagramm für Validierungs- und Testgenauigkeit.
    """

    percentages = [
        int(result["training_percentage"])
        for result in results
    ]

    validation_accuracies = [
        float(result["best_validation_accuracy"])
        for result in results
    ]

    test_accuracies = [
        float(result["test_accuracy"])
        for result in results
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        percentages,
        validation_accuracies,
        marker="o",
        linewidth=2,
        label="Beste Validierungsgenauigkeit",
    )

    plt.plot(
        percentages,
        test_accuracies,
        marker="s",
        linewidth=2,
        label="Testgenauigkeit",
    )

    plt.title(
        "Einfluss der Trainingsdatengröße "
        "auf die Modellgenauigkeit"
    )
    plt.xlabel("Verwendeter Anteil der Trainingsdaten in Prozent")
    plt.ylabel("Genauigkeit")
    plt.xticks(percentages)
    plt.ylim(0.5, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    result_path = (
        RESULTS_DIR / "experiment_training_sizes.png"
    )

    plt.savefig(
        result_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Genauigkeitsdiagramm gespeichert: {result_path}")


def create_duration_plot(
    results: list[dict[str, float | int]],
) -> None:
    """
    Erstellt ein Diagramm für die Trainingsdauer.
    """

    percentages = [
        int(result["training_percentage"])
        for result in results
    ]

    durations = [
        float(result["training_duration_seconds"])
        for result in results
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        percentages,
        durations,
        marker="o",
        linewidth=2,
    )

    plt.title(
        "Trainingsdauer bei unterschiedlichen "
        "Trainingsdatengrößen"
    )
    plt.xlabel("Verwendeter Anteil der Trainingsdaten in Prozent")
    plt.ylabel("Trainingsdauer in Sekunden")
    plt.xticks(percentages)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    result_path = (
        RESULTS_DIR
        / "experiment_training_sizes_duration.png"
    )

    plt.savefig(
        result_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Zeitdiagramm gespeichert: {result_path}")


def main() -> None:
    """
    Führt das vollständige Trainingsgrößenexperiment aus.
    """

    create_output_directories()

    (
        x_train_full,
        y_train_full,
        x_test,
        y_test,
    ) = load_prepared_sequences()

    (
        x_train_base,
        x_validation,
        y_train_base,
        y_validation,
    ) = train_test_split(
        x_train_full,
        y_train_full,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    print("\nAufteilung für das Experiment")
    print("-----------------------------")
    print(f"Trainingspool:  {x_train_base.shape}")
    print(f"Validierungsset: {x_validation.shape}")
    print(f"Testset:         {x_test.shape}")

    results = run_training_size_experiment(
        x_train_base=x_train_base,
        y_train_base=y_train_base,
        x_validation=x_validation,
        y_validation=y_validation,
        x_test=x_test,
        y_test=y_test,
    )

    save_results_as_json(results)
    save_results_as_csv(results)

    create_accuracy_plot(results)
    create_duration_plot(results)

    print(
        "\nExperiment zur Trainingsdatengröße "
        "erfolgreich abgeschlossen."
    )


if __name__ == "__main__":
    main()