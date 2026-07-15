"""Analyse und Vereinheitlichung der Sequenzlängen im IMDB-Datensatz."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

from config import (
    MAX_SEQUENCE_LENGTH,
    OUTPUT_DIR,
    RESULTS_DIR,
    create_output_directories,
)
from data_utils import ImdbDataset, load_imdb_dataset


def calculate_length_statistics(
    sequences: np.ndarray,
) -> dict[str, float]:
    """
    Berechnet grundlegende Statistiken zu den Sequenzlängen.
    """

    lengths = np.array(
        [len(sequence) for sequence in sequences],
        dtype=np.int32,
    )

    return {
        "minimum": int(np.min(lengths)),
        "maximum": int(np.max(lengths)),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "percentile_90": float(np.percentile(lengths, 90)),
        "percentile_95": float(np.percentile(lengths, 95)),
    }


def prepare_padded_sequences(
    dataset: ImdbDataset,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Bringt alle Trainings- und Testsequenzen auf dieselbe Länge.

    Kürzere Sequenzen werden mit dem PAD-Wert 0 ergänzt.
    Längere Sequenzen werden auf MAX_SEQUENCE_LENGTH gekürzt.
    """

    x_train_padded = pad_sequences(
        dataset.x_train,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        value=0,
    )

    x_test_padded = pad_sequences(
        dataset.x_test,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
        value=0,
    )

    return x_train_padded, x_test_padded


def calculate_truncation_share(
    sequences: np.ndarray,
) -> float:
    """
    Berechnet den Anteil der Sequenzen, die gekürzt werden.
    """

    number_of_long_sequences = sum(
        len(sequence) > MAX_SEQUENCE_LENGTH
        for sequence in sequences
    )

    return number_of_long_sequences / len(sequences)


def save_length_statistics(
    train_statistics: dict[str, float],
    test_statistics: dict[str, float],
    train_truncation_share: float,
    test_truncation_share: float,
) -> None:
    """
    Speichert die Längenstatistiken als JSON-Datei.
    """

    output_data = {
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "training_data": {
            **train_statistics,
            "truncation_share": train_truncation_share,
        },
        "test_data": {
            **test_statistics,
            "truncation_share": test_truncation_share,
        },
    }

    output_path = OUTPUT_DIR / "sequence_length_statistics.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            output_data,
            output_file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Längenstatistiken gespeichert: {output_path}")


def save_prepared_sequences(
    x_train_padded: np.ndarray,
    y_train: np.ndarray,
    x_test_padded: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """
    Speichert die vorbereiteten Sequenzen und Labels
    für weitere Experimente.
    """

    output_path = OUTPUT_DIR / "prepared_sequences.npz"

    np.savez_compressed(
        output_path,
        x_train=x_train_padded,
        y_train=y_train,
        x_test=x_test_padded,
        y_test=y_test,
    )

    print(f"Vorbereitete Sequenzen gespeichert: {output_path}")


def create_length_distribution_plot(
    sequences: np.ndarray,
) -> None:
    """
    Erstellt ein Histogramm der Sequenzlängen.
    """

    lengths = np.array(
        [len(sequence) for sequence in sequences],
        dtype=np.int32,
    )

    plt.figure(figsize=(10, 6))

    plt.hist(
        lengths,
        bins=50,
    )

    plt.axvline(
        MAX_SEQUENCE_LENGTH,
        linestyle="--",
        label=(
            f"Gewählte Maximallänge: "
            f"{MAX_SEQUENCE_LENGTH} Token"
        ),
    )

    plt.title("Verteilung der Längen von IMDB-Rezensionen")
    plt.xlabel("Anzahl der Token pro Rezension")
    plt.ylabel("Anzahl der Rezensionen")
    plt.legend()
    plt.tight_layout()

    result_path = (
        RESULTS_DIR / "sequence_length_distribution.png"
    )

    plt.savefig(
        result_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Diagramm gespeichert: {result_path}")


def print_statistics(
    title: str,
    statistics: dict[str, float],
    truncation_share: float,
) -> None:
    """
    Gibt die Längenstatistiken verständlich aus.
    """

    print(f"\n{title}")
    print("-" * len(title))
    print(f"Minimale Länge:      {statistics['minimum']:.0f}")
    print(f"Maximale Länge:      {statistics['maximum']:.0f}")
    print(f"Durchschnitt:         {statistics['mean']:.2f}")
    print(f"Median:               {statistics['median']:.2f}")
    print(
        f"90. Perzentil:        "
        f"{statistics['percentile_90']:.2f}"
    )
    print(
        f"95. Perzentil:        "
        f"{statistics['percentile_95']:.2f}"
    )
    print(
        "Anteil gekürzter Rezensionen: "
        f"{truncation_share * 100:.2f} %"
    )


def main() -> None:
    """
    Analysiert und vereinheitlicht die Sequenzlängen.
    """

    create_output_directories()

    dataset = load_imdb_dataset()

    train_statistics = calculate_length_statistics(
        dataset.x_train
    )

    test_statistics = calculate_length_statistics(
        dataset.x_test
    )

    train_truncation_share = calculate_truncation_share(
        dataset.x_train
    )

    test_truncation_share = calculate_truncation_share(
        dataset.x_test
    )

    print_statistics(
        title="Trainingsdaten",
        statistics=train_statistics,
        truncation_share=train_truncation_share,
    )

    print_statistics(
        title="Testdaten",
        statistics=test_statistics,
        truncation_share=test_truncation_share,
    )

    x_train_padded, x_test_padded = prepare_padded_sequences(
        dataset
    )

    print("\nForm nach dem Padding")
    print("---------------------")
    print(f"x_train_padded: {x_train_padded.shape}")
    print(f"x_test_padded:  {x_test_padded.shape}")

    print("\nBeispiel einer vorbereiteten Sequenz")
    print("------------------------------------")
    print(x_train_padded[0][:50])

    save_prepared_sequences(
        x_train_padded=x_train_padded,
        y_train=dataset.y_train,
        x_test_padded=x_test_padded,
        y_test=dataset.y_test,
    )

    save_length_statistics(
        train_statistics=train_statistics,
        test_statistics=test_statistics,
        train_truncation_share=train_truncation_share,
        test_truncation_share=test_truncation_share,
    )

    create_length_distribution_plot(
        dataset.x_train
    )


if __name__ == "__main__":
    main()