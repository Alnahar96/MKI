import math
import pandas as pd


# Aufgabe 3: k-Means 
# Daten aus Tabelle 


data = [
    {"Nr": 1, "Feuchte": "trocken", "Saeure": "basisch",   "Temp": 7,  "Wachstum": "gut"},
    {"Nr": 2, "Feuchte": "feucht",  "Saeure": "neutral",   "Temp": 8,  "Wachstum": "schlecht"},
    {"Nr": 3, "Feuchte": "trocken", "Saeure": "neutral",   "Temp": 7,  "Wachstum": "gut"},
    {"Nr": 4, "Feuchte": "feucht",  "Saeure": "alkalisch", "Temp": 5,  "Wachstum": "schlecht"},
    {"Nr": 5, "Feuchte": "trocken", "Saeure": "neutral",   "Temp": 8,  "Wachstum": "schlecht"},
    {"Nr": 6, "Feuchte": "trocken", "Saeure": "neutral",   "Temp": 6,  "Wachstum": "gut"},
    {"Nr": 7, "Feuchte": "trocken", "Saeure": "neutral",   "Temp": 11, "Wachstum": "schlecht"},
    {"Nr": 8, "Feuchte": "trocken", "Saeure": "neutral",   "Temp": 9,  "Wachstum": "schlecht"},
    {"Nr": 9, "Feuchte": "trocken", "Saeure": "alkalisch", "Temp": 9,  "Wachstum": "gut"},
    {"Nr": 10,"Feuchte": "trocken", "Saeure": "alkalisch", "Temp": 8,  "Wachstum": "gut"},
    {"Nr": 11,"Feuchte": "feucht",  "Saeure": "basisch",   "Temp": 7,  "Wachstum": "schlecht"},
    {"Nr": 12,"Feuchte": "feucht",  "Saeure": "neutral",   "Temp": 10, "Wachstum": "gut"},
    {"Nr": 13,"Feuchte": "trocken", "Saeure": "basisch",   "Temp": 6,  "Wachstum": "gut"},
    {"Nr": 14,"Feuchte": "feucht",  "Saeure": "alkalisch", "Temp": 7,  "Wachstum": "schlecht"},
    {"Nr": 15,"Feuchte": "trocken", "Saeure": "basisch",   "Temp": 3,  "Wachstum": "schlecht"},
    {"Nr": 16,"Feuchte": "trocken", "Saeure": "basisch",   "Temp": 4,  "Wachstum": "gut"},
]

df = pd.DataFrame(data)

# ------------------------------------------------------------
#  Kodierung der kategorialen Merkmale


feuchte_mapping = {
    "trocken": 0,
    "feucht": 1
}

saeure_mapping = {
    "basisch": 0,
    "neutral": 1,
    "alkalisch": 2
}

df["Feuchte_code"] = df["Feuchte"].map(feuchte_mapping)
df["Saeure_code"] = df["Saeure"].map(saeure_mapping)


features = ["Feuchte_code", "Saeure_code", "Temp"]

print("\nKodierte Daten:")
print(df[["Nr", "Feuchte", "Saeure", "Temp", "Wachstum", "Feuchte_code", "Saeure_code"]])

# ------------------------------------------------------------
#  Hilfsfunktionen


def euclidean_distance(point_a, point_b):
   #distance 
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def calculate_centroid(points):
    
    # Mittelpunkt eines Clusters.
    
    number_of_points = len(points)   # pointe in cluster
    number_of_features = len(points[0])   # features pro point

    centroid = []
    for feature_index in range(number_of_features):     #  add the feature value von allen points in cluster und teile durch anzahl der points
        feature_sum = sum(point[feature_index] for point in points)
        centroid.append(feature_sum / number_of_points)

    return centroid


# ------------------------------------------------------------
#  k-Means mit k = 2
# Initiale Zentren: DP1 und DP8


training_df = df[df["Nr"] <= 10].copy()

points = {
    row["Nr"]: [row["Feuchte_code"], row["Saeure_code"], row["Temp"]]
    for _, row in training_df.iterrows()
}

centroid_1 = points[1]  # DP1
centroid_2 = points[8]  # DP8

print("\nInitiale Clusterzentren:")
print(f"C1 = DP1 = {centroid_1}")
print(f"C2 = DP8 = {centroid_2}")

previous_clusters = None

for iteration in range(1, 10):
    cluster_1 = []
    cluster_2 = []

    print(f"\nIteration {iteration}:")

    for nr, point in points.items():
        distance_to_c1 = euclidean_distance(point, centroid_1)
        distance_to_c2 = euclidean_distance(point, centroid_2)

        if distance_to_c1 <= distance_to_c2:
            cluster_1.append(nr)
            assigned_cluster = "Cluster 1"
        else:
            cluster_2.append(nr)
            assigned_cluster = "Cluster 2"

        print(
            f"DP{nr}: Punkt={point}, "
            f"d(C1)={distance_to_c1:.2f}, "
            f"d(C2)={distance_to_c2:.2f} -> {assigned_cluster}"
        )

    current_clusters = (cluster_1, cluster_2)

    print(f"\nCluster 1: {cluster_1}")
    print(f"Cluster 2: {cluster_2}")

    if current_clusters == previous_clusters:
        print("\nKeine Änderung der Cluster. k-Means ist konvergiert.")
        break

    centroid_1 = calculate_centroid([points[nr] for nr in cluster_1])
    centroid_2 = calculate_centroid([points[nr] for nr in cluster_2])

    print(f"Neues Zentrum C1 = {[round(x, 2) for x in centroid_1]}")
    print(f"Neues Zentrum C2 = {[round(x, 2) for x in centroid_2]}")

    previous_clusters = current_clusters


# ------------------------------------------------------------
# 4. Zuordnung von DP11 und DP16
# ------------------------------------------------------------

print("\nFinale Clusterzentren:")
print(f"C1 = {[round(x, 2) for x in centroid_1]}")
print(f"C2 = {[round(x, 2) for x in centroid_2]}")

test_points = df[df["Nr"].isin([11, 16])]

print("\nZuordnung von DP11 und DP16:")

for _, row in test_points.iterrows():
    nr = row["Nr"]
    point = [row["Feuchte_code"], row["Saeure_code"], row["Temp"]]

    distance_to_c1 = euclidean_distance(point, centroid_1)
    distance_to_c2 = euclidean_distance(point, centroid_2)

    if distance_to_c1 <= distance_to_c2:
        assigned_cluster = "Cluster 1"
    else:
        assigned_cluster = "Cluster 2"

    print(
        f"DP{nr}: Punkt={point}, "
        f"d(C1)={distance_to_c1:.2f}, "
        f"d(C2)={distance_to_c2:.2f} -> {assigned_cluster}"
    )