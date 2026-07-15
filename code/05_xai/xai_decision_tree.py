from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


RANDOM_SEED = 42
TEST_SIZE = 0.20
MAX_DEPTH = 3


def create_directories(base_dir: Path) -> tuple[Path, Path]:
    """Create output directories and return their paths."""
    results_dir = base_dir / "results"
    outputs_dir = base_dir / "outputs"

    results_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    return results_dir, outputs_dir


def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the Iris dataset and return features, labels, and class names."""
    iris = load_iris(as_frame=True)

    features = iris.data
    labels = iris.target
    class_names = list(iris.target_names)

    return features, labels, class_names


def train_model(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[
    DecisionTreeClassifier,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split the data and train an interpretable decision tree."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    model = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        random_state=RANDOM_SEED,
    )

    model.fit(x_train, y_train)

    return model, x_train, x_test, y_train, y_test


def save_tree_visualization(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Visualize and save the trained decision tree."""
    plt.figure(figsize=(18, 10))

    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        proportion=True,
        precision=2,
    )

    plt.title("Visualisierung des Decision Trees")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_importance(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """Create and save a feature-importance chart."""
    importance_df = pd.DataFrame(
        {
            "Merkmal": feature_names,
            "Bedeutung": model.feature_importances_,
        }
    ).sort_values("Bedeutung", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(
        importance_df["Merkmal"],
        importance_df["Bedeutung"],
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Merkmal")
    plt.title("Bedeutung der Merkmale für den Decision Tree")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return importance_df.sort_values("Bedeutung", ascending=False)


def explain_single_prediction(
    model: DecisionTreeClassifier,
    sample: pd.DataFrame,
    feature_names: list[str],
) -> list[str]:
    """Extract the decision path for one prediction."""
    node_indicator = model.decision_path(sample)
    leaf_id = model.apply(sample)[0]
    tree = model.tree_

    explanation_steps: list[str] = []

    for node_id in node_indicator.indices:
        if node_id == leaf_id:
            explanation_steps.append(
                f"Blattknoten {node_id}: endgültige Klassifikation erreicht."
            )
            continue

        feature_index = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_index]
        feature_value = sample.iloc[0, feature_index]

        if feature_value <= threshold:
            operator = "<="
        else:
            operator = ">"

        explanation_steps.append(
            f"Knoten {node_id}: {feature_name} = {feature_value:.2f} "
            f"{operator} {threshold:.2f}"
        )

    return explanation_steps


def save_results(
    model: DecisionTreeClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    class_names: list[str],
    importance_df: pd.DataFrame,
    explanation_steps: list[str],
    output_path: Path,
) -> None:
    """Save metrics, feature importance, and one local explanation."""
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    sample = x_test.iloc[[0]]
    predicted_class = int(model.predict(sample)[0])
    actual_class = int(y_test.iloc[0])

    with output_path.open("w", encoding="utf-8") as file:
        file.write("XAI-Experiment mit Decision Tree\n")
        file.write("=" * 40 + "\n\n")

        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Maximale Baumtiefe: {MAX_DEPTH}\n")
        file.write(f"Testanteil: {TEST_SIZE:.0%}\n")
        file.write(f"Random Seed: {RANDOM_SEED}\n\n")

        file.write("Klassifikationsbericht:\n")
        file.write(
            classification_report(
                y_test,
                predictions,
                target_names=class_names,
            )
        )

        file.write("\nFeature Importance:\n")
        file.write(importance_df.to_string(index=False))
        file.write("\n\n")

        file.write("Beispiel einer lokalen Erklärung:\n")
        file.write(sample.to_string(index=False))
        file.write("\n\n")
        file.write(
            f"Tatsächliche Klasse: {class_names[actual_class]}\n"
        )
        file.write(
            f"Vorhergesagte Klasse: {class_names[predicted_class]}\n\n"
        )

        for step in explanation_steps:
            file.write(f"- {step}\n")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    results_dir, outputs_dir = create_directories(base_dir)

    features, labels, class_names = load_dataset()

    model, _, x_test, _, y_test = train_model(
        features,
        labels,
    )

    tree_path = results_dir / "decision_tree.png"
    importance_path = results_dir / "feature_importance.png"
    output_path = outputs_dir / "xai_results.txt"

    save_tree_visualization(
        model=model,
        feature_names=list(features.columns),
        class_names=class_names,
        output_path=tree_path,
    )

    importance_df = save_feature_importance(
        model=model,
        feature_names=list(features.columns),
        output_path=importance_path,
    )

    sample = x_test.iloc[[0]]

    explanation_steps = explain_single_prediction(
        model=model,
        sample=sample,
        feature_names=list(features.columns),
    )

    save_results(
        model=model,
        x_test=x_test,
        y_test=y_test,
        class_names=class_names,
        importance_df=importance_df,
        explanation_steps=explanation_steps,
        output_path=output_path,
    )

    print("XAI-Experiment erfolgreich abgeschlossen.")
    print(f"Entscheidungsbaum: {tree_path}")
    print(f"Feature Importance: {importance_path}")
    print(f"Ergebnisse: {output_path}")


if __name__ == "__main__":
    main()