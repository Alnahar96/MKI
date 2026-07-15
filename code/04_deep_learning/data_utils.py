"""Hilfsfunktionen zum Laden und Untersuchen des IMDB-Datensatzes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tensorflow.keras.datasets import imdb

from config import RANDOM_SEED, VOCAB_SIZE


@dataclass
class ImdbDataset:
    """Container für Trainings- und Testdaten."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def load_imdb_dataset() -> ImdbDataset:
    """
    Lädt den IMDB-Datensatz mit begrenztem Vokabular. cdwc

    Es werden nur die häufigsten Wörter bis zur festgelegten
    VOCAB_SIZE berücksichtigt.
    """

    np.random.seed(RANDOM_SEED)

    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=VOCAB_SIZE
    )

    return ImdbDataset(
        x_train=np.array(x_train, dtype=object),
        y_train=np.asarray(y_train, dtype=np.int32),
        x_test=np.array(x_test, dtype=object),
        y_test=np.asarray(y_test, dtype=np.int32),
    )


def load_word_mappings() -> tuple[dict[str, int], dict[int, str]]:
    """
    Lädt das Keras-Wörterbuch und erstellt beide Zuordnungsrichtungen.

    Rückgabe:
    - word_to_id: Wort -> Token-ID
    - id_to_word: Token-ID -> Wort
    """

    original_word_to_id = imdb.get_word_index()

    # Keras reserviert die ersten IDs für spezielle Tokens.
    word_to_id = {
        word: token_id + 3
        for word, token_id in original_word_to_id.items()
    }

    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3

    id_to_word = {
        token_id: word
        for word, token_id in word_to_id.items()
    }

    return word_to_id, id_to_word


def decode_review(
    encoded_review: list[int] | np.ndarray,
    id_to_word: dict[int, str],
) -> str:
    """
    Wandelt eine Sequenz von Token-IDs zurück in lesbaren Text um.
    """

    return " ".join(
        id_to_word.get(int(token_id), "<UNK>")
        for token_id in encoded_review
    )


def label_to_text(label: int) -> str:
    """Wandelt das binäre Label in eine lesbare Bezeichnung um."""

    if label == 1:
        return "positiv"

    if label == 0:
        return "negativ"

    raise ValueError(f"Unbekanntes Label: {label}")


def print_dataset_summary(dataset: ImdbDataset) -> None:
    """Gibt zentrale Informationen zum Datensatz aus."""

    train_positive = int(np.sum(dataset.y_train == 1))
    train_negative = int(np.sum(dataset.y_train == 0))
    test_positive = int(np.sum(dataset.y_test == 1))
    test_negative = int(np.sum(dataset.y_test == 0))

    print("\nDatensatzübersicht")
    print("------------------")
    print(f"Trainingsbeispiele: {len(dataset.x_train)}")
    print(f"Testbeispiele:      {len(dataset.x_test)}")
    print(f"Positive Trainingsbeispiele: {train_positive}")
    print(f"Negative Trainingsbeispiele: {train_negative}")
    print(f"Positive Testbeispiele:      {test_positive}")
    print(f"Negative Testbeispiele:      {test_negative}")


def print_review_examples(
    dataset: ImdbDataset,
    id_to_word: dict[int, str],
    number_of_examples: int = 3,
) -> None:
    """Zeigt einige decodierte Filmkritiken mit ihren Labels."""

    if number_of_examples <= 0:
        raise ValueError("number_of_examples muss größer als 0 sein.")

    number_of_examples = min(number_of_examples, len(dataset.x_train))

    print("\nBeispielrezensionen")
    print("-------------------")

    for index in range(number_of_examples):
        decoded_text = decode_review(
            dataset.x_train[index],
            id_to_word,
        )

        print(f"\nBeispiel {index + 1}")
        print(f"Label: {label_to_text(int(dataset.y_train[index]))}")
        print(f"Anzahl Token-IDs: {len(dataset.x_train[index])}")
        print(decoded_text[:1000])


def main() -> None:
    """Lädt den Datensatz und zeigt eine erste Analyse."""

    dataset = load_imdb_dataset()
    _, id_to_word = load_word_mappings()

    print_dataset_summary(dataset)
    print_review_examples(
        dataset=dataset,
        id_to_word=id_to_word,
        number_of_examples=3,
    )


if __name__ == "__main__":
    main()