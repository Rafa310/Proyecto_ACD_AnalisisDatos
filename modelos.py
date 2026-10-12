import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, recall_score, f1_score)

os.makedirs("figuras", exist_ok=True)

# --- Datos ---
df = pd.read_csv("data/data.csv", sep=None, engine="python", encoding="utf-8-sig")
df.columns = df.columns.str.strip()
df = df.drop_duplicates()

y = (df["Target"] == "Dropout").astype(int)   # 1 = abandona, 0 = no abandona
X = df.drop(columns=["Target"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Modelos (los que usan distancias/pesos llevan escalado) ---
modelos = {
    "Árbol de decisión": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
    "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Red neuronal": make_pipeline(StandardScaler(),
                                  MLPClassifier(hidden_layer_sizes=(32, 16),
                                                max_iter=1500, random_state=42)),
}

resultados = []
predicciones = {}

for nombre, modelo in modelos.items():
    cv = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1").mean()
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    predicciones[nombre] = pred
    resultados.append({
        "Modelo": nombre,
        "Accuracy": accuracy_score(y_test, pred),
        "Recall (dropout)": recall_score(y_test, pred),
        "F1 (dropout)": f1_score(y_test, pred),
        "F1 validación cruzada": cv,
    })
    print(f"\n===== {nombre} =====")
    print(classification_report(y_test, pred, target_names=["No abandona", "Abandona"]))

tabla = pd.DataFrame(resultados).sort_values("F1 (dropout)", ascending=False).round(3)
print("\n=== COMPARACIÓN DE MODELOS ===")
print(tabla.to_string(index=False))
tabla.to_csv("resultados_modelos.csv", index=False)

# --- Matriz de confusión del mejor modelo ---
mejor = tabla.iloc[0]["Modelo"]
cm = confusion_matrix(y_test, predicciones[mejor])
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No abandona", "Abandona"],
            yticklabels=["No abandona", "Abandona"])
plt.title(f"Matriz de confusión - {mejor}")
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("figuras/05_matriz_confusion.png")
plt.show()

# --- Variables más importantes (árbol de decisión) ---
arbol = modelos["Árbol de decisión"]
imp = pd.Series(arbol.feature_importances_, index=X.columns).sort_values().tail(10)
plt.figure(figsize=(8, 5))
imp.plot(kind="barh")
plt.title("Top 10 variables más importantes (árbol de decisión)")
plt.tight_layout()
plt.savefig("figuras/06_importancia_variables.png")
plt.show()