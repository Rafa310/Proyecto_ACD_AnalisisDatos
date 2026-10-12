# Procedencia de los datos.
Los datos fueron estraídos de un DataSet encontrado en UCI:
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Predicción de deserción universitaria

Proyecto de cátedra de Aprendizaje Automático y Ciencia de Datos.

Se desarrolla un modelo de machine learning supervisado que predice si un estudiante universitario abandonará sus estudios, usando datos académicos, demográficos y socioeconómicos.

Integrantes
Angel Eduardo moreno Escobar ME220001
Edwin Antonio Ticas Romero TR220316 
Rafael Adolfo Ruiz García RG210380
Ricardo Iván Escobar Umaña EU220488

Planteamiento del problema

La deserción universitaria afecta a los estudiantes, a sus familias y a las instituciones. Detectarla a tiempo permite ofrecer apoyo (tutorías, becas, acompañamiento) antes de que el estudiante abandone.
Pregunta de investigación: ¿Es posible identificar tempranamente a los estudiantes en riesgo de abandonar la universidad a partir de sus datos académicos y socioeconómicos?


# Objetivo General:

# Objetivos Específicos:

- Objetivo 1
- Objetivo 2
- Objetivo 3

Los datos provienen del repositorio UCI Machine Learning Repository: 
https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

Nombre: Predict Students' Dropout and Academic Success
Autores: Realinho, V., Vieira Martins, M., Machado, J. y Baptista, L. (2021)
Origen: Instituto Politécnico de Portalegre (Portugal)
Tamaño: 4,424 estudiantes y 36 variables, sin valores nulos ni duplicados
Variable objetivo (Target): Dropout, Enrolled o Graduate

El conjunto de datos está anonimizado y no contiene datos personales identificables.

# Estructura del proyecto
´´´
ProyectoCatedraACD/
├── data/
│   └── data.csv              # Dataset original
├── figuras/                  # Gráficas generadas por los scripts
├── exploracion.py            # Análisis exploratorio de datos (EDA)
├── modelos.py                # Entrenamiento y evaluación de modelos
├── resultados_modelos.csv    # Tabla comparativa de modelos
├── requirements.txt          # Dependencias
└── README.md
´´´

# Cómo ejecutar el proyecto
1) Clonar el repositorio y entrar a la carpeta del proyecto.
2) Crear y activar un entorno virtual (se recomienda Python 3.12 o 3.13).
3) Instalar las dependencias:
´´´
pip install -r requirements.txt
´´´
4) Verificar que el dataset esté en data/data.csv.
5) Ejecutar los scripts en este orden:
´´´
python exploracion.py 
python modelos.py
´´´
# Metodología
Carga y revisión: dimensiones, tipos de datos, nulos y duplicados.
Análisis exploratorio: distribución de la variable objetivo y relación del abandono con la beca, las notas y la edad.
Preparación: la variable objetivo se convierte en binaria (1 = abandona, 0 = no abandona; "Enrolled" y "Graduate" se agrupan como "no abandona").
Partición: 80% entrenamiento y 20% prueba, estratificada para conservar la proporción de clases.
Modelado: cinco algoritmos con validación cruzada de 5 pliegues. Los modelos sensibles a la escala (kNN, regresión logística, red neuronal) usan StandardScaler.
Evaluación: se prioriza el recall y el F1 de la clase "abandona", porque no detectar a un estudiante en riesgo es más costoso que una falsa alarma.

Los resultados son estos:
| Modelo | Accuracy | Recall (abandona) | F1 (abandona) | F1 val. cruzada |
|---|---|---|---|---|
| Regresión logística | 0.886 | 0.736 | 0.805 | 0.781 |
| Red neuronal | 0.849 | 0.750 | 0.761 | 0.723 |
| Árbol de decisión | 0.854 | 0.690 | 0.752 | 0.756 |
| Naive Bayes | 0.826 | 0.718 | 0.726 | 0.720 |
| kNN | 0.829 | 0.567 | 0.681 | 0.696 |


# Consideraciones éticas y legales:

# Tecnologías:

Python, Pandas, NumPy, Matplotlib, Seaborn y scikit-learn.