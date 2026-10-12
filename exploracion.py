import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("figuras", exist_ok=True)

# sep=None detecta solo si el separador es ; o ,
df = pd.read_csv("data/data.csv", sep=None, engine="python", encoding="utf-8-sig")
df.columns = df.columns.str.strip()  # limpia espacios/tabs en los nombres

print("Dimensiones:", df.shape)
print("\nPrimeras filas:\n", df.head())
print("\nInfo:")
df.info()
print("\nNulos por columna:\n", df.isnull().sum().sum(), "nulos en total")
print("Duplicados:", df.duplicated().sum())
print("\nDistribución de la variable objetivo:\n", df["Target"].value_counts())

# 1. Distribución de la clase
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Target", order=["Dropout", "Enrolled", "Graduate"])
plt.title("Distribución de estudiantes por estado")
plt.tight_layout()
plt.savefig("figuras/01_distribucion_target.png")
plt.show()

# 2. Deserción según beca
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Scholarship holder", hue="Target")
plt.title("Estado del estudiante según beca (0 = no, 1 = sí)")
plt.tight_layout()
plt.savefig("figuras/02_beca_vs_target.png")
plt.show()

# 3. Notas del primer semestre
plt.figure(figsize=(6, 4))
sns.boxplot(data=df, x="Target", y="Curricular units 1st sem (grade)",
            order=["Dropout", "Enrolled", "Graduate"])
plt.title("Nota promedio 1er semestre por estado")
plt.tight_layout()
plt.savefig("figuras/03_notas_vs_target.png")
plt.show()

# 4. Edad al matricularse
plt.figure(figsize=(7, 4))
sns.histplot(data=df, x="Age at enrollment", hue="Target", bins=25, multiple="stack")
plt.title("Edad al matricularse")
plt.tight_layout()
plt.savefig("figuras/04_edad.png")
plt.show()