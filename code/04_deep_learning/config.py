"""Zentrale Konfiguration für das Deep-Learning-NLP-Experiment."""

from pathlib import Path

RANDOM_SEED = 42

VOCAB_SIZE = 10_000
MAX_SEQUENCE_LENGTH = 250

EMBEDDING_DIM = 32
HIDDEN_UNITS = 16
DROPOUT_RATE = 0.3

BATCH_SIZE = 128
EPOCHS = 10

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
RESULTS_DIR = PROJECT_DIR / "results"


def create_output_directories() -> None:
    """Erstellt die benötigten Ausgabeordner, falls sie noch nicht existieren."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_output_directories()

    print("Konfiguration erfolgreich geladen.")
    print(f"Projektordner: {PROJECT_DIR}")
    print(f"Ausgabeordner: {OUTPUT_DIR}")
    print(f"Ergebnisordner: {RESULTS_DIR}")