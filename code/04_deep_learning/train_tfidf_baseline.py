"""Training und Bewertung eines TF-IDF-Baseline-Modells."""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import (
    OUTPUT_DIR,
    RANDOM_SEED,
    RESULTS_DIR,
    VOCAB_SIZE,
    create_output_directories,
)
from data_utils import (
    decode_review,
    load_imdb_dataset,
    load_word_mappings,
)


def decode_reviews(
    encoded_reviews: np.ndarray,
    id_to_word: dict[int, str],
) -> list[str]:
    """
    Wandelt mehrere Sequenzen von Token-IDs in lesbare Texte um.

    Spezielle Tokens werden entfernt, da sie für das TF-IDF-Modell
    keine inhaltliche Bedeutung besitzen.
    """

    decoded_reviews: list[str] = []

    for encoded_review in encoded_reviews:
        decoded_text = decode_review(
            encoded_review=encoded_review,
            id_to_word=id_to_word,
        )

        cleaned_text = (
            decoded_text
            .replace("<START>", " ")
            .replace("<PAD>", " ")
            .replace("<UNUSED>", " ")
        )

        decoded_reviews.append(
            " ".join(cleaned_text.split())
        )

    return decoded_reviews


def create_vectorizer() -> TfidfVectorizer:
    """
    Erstellt die TF-IDF-Repräsentation.

    Die Repräsentation ist dünn besetzt und häufigkeitsbasiert.
    Sie lernt keine dichten semantischen Wortvektoren.
    """

    return TfidfVectorizer(
        max_features=VOCAB_SIZE,
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
    )


def create_classifier() -> LogisticRegression:
    """Erstellt den Klassifikator für die binäre Sentimentanalyse."""

    return LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        random_state=RANDOM_SEED,
    )


def calculate_metrics(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> dict[str, float]:
    """Berechnet die wichtigsten Klassifikationsmetriken."""

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


def save_metrics(
    metrics: dict[str, float],
) -> None:
    """Speichert die Modellmetriken als JSON-Datei."""

    output_path = OUTPUT_DIR / "metrics_tfidf.json"

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


def save_predictions(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    positive_probabilities: np.ndarray,
) -> None:
    """
    Speichert Vorhersagen für die spätere Fehleranalyse.
    """

    output_path = OUTPUT_DIR / "tfidf_predictions.npz"

    np.savez_compressed(
        output_path,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        positive_probabilities=positive_probabilities,
    )

    print(f"Vorhersagen gespeichert: {output_path}")


def create_confusion_matrix_plot(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> None:
    """Erstellt eine Confusion Matrix für das Baseline-Modell."""

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["negativ", "positiv"],
    )

    display.plot(
        values_format="d",
    )

    plt.title(
        "Konfusionsmatrix: TF-IDF und logistische Regression"
    )
    plt.xlabel("Vorhergesagte Klasse")
    plt.ylabel("Tatsächliche Klasse")
    plt.tight_layout()

    result_path = (
        RESULTS_DIR / "confusion_matrix_tfidf.png"
    )

    plt.savefig(
        result_path,
        dpi=300,
    )
    plt.close()

    print(f"Konfusionsmatrix gespeichert: {result_path}")


def print_metrics(
    metrics: dict[str, float],
) -> None:
    """Gibt die Ergebnisse verständlich im Terminal aus."""

    print("\nErgebnisse des TF-IDF-Baseline-Modells")
    print("--------------------------------------")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision']:.4f}")
    print(f"Recall:         {metrics['recall']:.4f}")
    print(f"F1-Score:       {metrics['f1_score']:.4f}")
    print(
        f"Trainingszeit:  "
        f"{metrics['training_time_seconds']:.2f} Sekunden"
    )
    print(
        f"Anzahl Merkmale: "
        f"{int(metrics['number_of_features'])}"
    )


def main() -> None:
    """Trainiert und bewertet das TF-IDF-Baseline-Modell."""

    create_output_directories()

    print("IMDB-Datensatz wird geladen ...")
    dataset = load_imdb_dataset()

    print("Wörterbuch wird geladen ...")
    _, id_to_word = load_word_mappings()

    print("Trainingsrezensionen werden decodiert ...")
    train_texts = decode_reviews(
        encoded_reviews=dataset.x_train,
        id_to_word=id_to_word,
    )

    print("Testrezensionen werden decodiert ...")
    test_texts = decode_reviews(
        encoded_reviews=dataset.x_test,
        id_to_word=id_to_word,
    )

    vectorizer = create_vectorizer()

    print("TF-IDF-Merkmale werden erstellt ...")
    x_train_tfidf = vectorizer.fit_transform(
        train_texts
    )

    # Nur transform(): Das Testset darf das Vokabular
    # und die IDF-Gewichte nicht beeinflussen.
    x_test_tfidf = vectorizer.transform(
        test_texts
    )

    classifier = create_classifier()

    print("Logistische Regression wird trainiert ...")
    training_start = time.perf_counter()

    classifier.fit(
        x_train_tfidf,
        dataset.y_train,
    )

    training_time = (
        time.perf_counter() - training_start
    )

    print("Testdaten werden klassifiziert ...")
    predicted_labels = classifier.predict(
        x_test_tfidf
    )

    positive_probabilities = classifier.predict_proba(
        x_test_tfidf
    )[:, 1]

    metrics = calculate_metrics(
        true_labels=dataset.y_test,
        predicted_labels=predicted_labels,
    )

    metrics.update(
        {
            "training_time_seconds": float(training_time),
            "number_of_features": int(
                x_train_tfidf.shape[1]
            ),
            "number_of_training_examples": int(
                x_train_tfidf.shape[0]
            ),
            "number_of_test_examples": int(
                x_test_tfidf.shape[0]
            ),
            "random_seed": RANDOM_SEED,
        }
    )

    print_metrics(metrics)
    save_metrics(metrics)

    save_predictions(
        true_labels=dataset.y_test,
        predicted_labels=predicted_labels,
        positive_probabilities=positive_probabilities,
    )

    create_confusion_matrix_plot(
        true_labels=dataset.y_test,
        predicted_labels=predicted_labels,
    )


if __name__ == "__main__":
    main()