import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Ordner für Ergebnisse erstellen
output_dir = "code/01_data_mining"
os.makedirs(output_dir, exist_ok=True)

# 1. Datensatz laden
iris = load_iris(as_frame=True)
df = iris.frame

# 2. Erste Übersicht
print("Erste fünf Zeilen:")
print(df.head())

print("\nInformationen zum Datensatz:")
print(df.info())

print("\nStatistische Beschreibung:")
print(df.describe())

print("\nFehlende Werte:")
print(df.isnull().sum())

# 3. Zielklassen verständlicher machen
df["target_name"] = df["target"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

# 4. Visualisierung: Zusammenhang zwischen zwei Merkmalen
plt.figure(figsize=(8, 6))

for species in df["target_name"].unique():
    subset = df[df["target_name"] == species]
    plt.scatter(
        subset["petal length (cm)"],
        subset["petal width (cm)"],
        label=species
    )

plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.title("Data Mining: Muster im Iris-Datensatz")
plt.legend()
plt.grid(True)
plt.tight_layout()

# 5. Grafik speichern
output_path = os.path.join(output_dir, "iris_scatterplot.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"\nDie Grafik wurde gespeichert unter: {output_path}")