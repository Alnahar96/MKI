"""Vergleich des TF-IDF-Modells mit dem neuronalen Embedding-Modell."""

from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt

from config import OUTPUT_DIR, RESULTS_DIR, create_output_directories


def load_metrics(filename: str) -> dict[str, float]:
    """Lädt eine gespeicherte JSON-Datei mit Modellmetriken."""

    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Die benötigte Datei wurde nicht gefunden: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def save_comparison_csv(
    tfidf_metrics: dict[str, float],
    embedding_metrics: dict[str, float],
) -> None:
    """Speichert die wichtigsten Ergebnisse in einer Vergleichstabelle."""

    output_path = OUTPUT_DIR / "model_comparison.csv"

    fieldnames = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "training_time_seconds",
    ]

    rows = [
        {
            "model": "TF-IDF + Logistische Regression",
            "accuracy": tfidf_metrics["accuracy"],
            "precision": tfidf_metrics["precision"],
            "recall": tfidf_metrics["recall"],
            "f1_score": tfidf_metrics["f1_score"],
            "training_time_seconds": tfidf_metrics[
                "training_time_seconds"
            ],
        },
        {
            "model": "Neuronales Embedding-Modell",
            "accuracy": embedding_metrics["accuracy"],
            "precision": embedding_metrics["precision"],
            "recall": embedding_metrics["recall"],
            "f1_score": embedding_metrics["f1_score"],
            "training_time_seconds": embedding_metrics[
                "training_time_seconds"
            ],
        },
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
        writer.writerows(rows)

    print(f"Vergleichstabelle gespeichert: {output_path}")


def create_metric_comparison_plot(
    tfidf_metrics: dict[str, float],
    embedding_metrics: dict[str, float],
) -> None:
    """Visualisiert die vier Klassifikationsmetriken beider Modelle."""

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
    ]

    tfidf_values = [
        tfidf_metrics["accuracy"],
        tfidf_metrics["precision"],
        tfidf_metrics["recall"],
        tfidf_metrics["f1_score"],
    ]

    embedding_values = [
        embedding_metrics["accuracy"],
        embedding_metrics["precision"],
        embedding_metrics["recall"],
        embedding_metrics["f1_score"],
    ]

    x_positions = list(range(len(metric_names)))
    bar_width = 0.35

    left_positions = [
        position - bar_width / 2
        for position in x_positions
    ]

    right_positions = [
        position + bar_width / 2
        for position in x_positions
    ]

    plt.figure(figsize=(10, 6))

    plt.bar(
        left_positions,
        tfidf_values,
        width=bar_width,
        label="TF-IDF + logistische Regression",
    )

    plt.bar(
        right_positions,
        embedding_values,
        width=bar_width,
        label="Neuronales Embedding-Modell",
    )

    plt.xticks(
        x_positions,
        metric_names,
    )

    plt.ylim(0.80, 0.92)
    plt.title("Vergleich der Klassifikationsleistung")
    plt.xlabel("Bewertungsmetrik")
    plt.ylabel("Metrikwert")
    plt.legend()
    plt.tight_layout()

    result_path = RESULTS_DIR / "model_metric_comparison.png"

    plt.savefig(
        result_path,
        dpi=300,
    )
    plt.close()

    print(f"Metrikvergleich gespeichert: {result_path}")


def create_training_time_plot(
    tfidf_metrics: dict[str, float],
    embedding_metrics: dict[str, float],
) -> None:
    """Vergleicht die gemessenen Trainingszeiten."""

    model_names = [
        "TF-IDF +\nlogistische Regression",
        "Embedding-\nModell",
    ]

    training_times = [
        tfidf_metrics["training_time_seconds"],
        embedding_metrics["training_time_seconds"],
    ]

    plt.figure(figsize=(8, 6))
    plt.bar(
        model_names,
        training_times,
    )

    plt.title("Vergleich der Trainingszeiten")
    plt.xlabel("Modell")
    plt.ylabel("Trainingszeit in Sekunden")
    plt.tight_layout()

    result_path = RESULTS_DIR / "training_time_comparison.png"

    plt.savefig(
        result_path,
        dpi=300,
    )
    plt.close()

    print(f"Zeitvergleich gespeichert: {result_path}")


def print_comparison(
    tfidf_metrics: dict[str, float],
    embedding_metrics: dict[str, float],
) -> None:
    """Gibt die wichtigsten Unterschiede aus."""

    accuracy_difference = (
        embedding_metrics["accuracy"]
        - tfidf_metrics["accuracy"]
    )

    f1_difference = (
        embedding_metrics["f1_score"]
        - tfidf_metrics["f1_score"]
    )

    time_factor = (
        embedding_metrics["training_time_seconds"]
        / tfidf_metrics["training_time_seconds"]
    )

    print("\nModellvergleich")
    print("----------------")
    print(
        "Accuracy-Differenz "
        "(Embedding minus TF-IDF): "
        f"{accuracy_difference:+.4f}"
    )
    print(
        "F1-Differenz "
        "(Embedding minus TF-IDF): "
        f"{f1_difference:+.4f}"
    )
    print(
        "Faktor der Trainingszeit "
        f"(Embedding / TF-IDF): {time_factor:.2f}"
    )


def main() -> None:
    """Erstellt Tabellen und Diagramme für den Modellvergleich."""

    create_output_directories()

    tfidf_metrics = load_metrics(
        "metrics_tfidf.json"
    )

    embedding_metrics = load_metrics(
        "metrics_embedding.json"
    )

    print_comparison(
        tfidf_metrics=tfidf_metrics,
        embedding_metrics=embedding_metrics,
    )

    save_comparison_csv(
        tfidf_metrics=tfidf_metrics,
        embedding_metrics=embedding_metrics,
    )

    create_metric_comparison_plot(
        tfidf_metrics=tfidf_metrics,
        embedding_metrics=embedding_metrics,
    )

    create_training_time_plot(
        tfidf_metrics=tfidf_metrics,
        embedding_metrics=embedding_metrics,
    )


if __name__ == "__main__":
    main()