import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

URL = r'C:\Users\yilena.mosquera\Downloads\Ejercicio_Datos.xlsx'
data = pd.read_excel(URL)

data = pd.get_dummies(data, columns=['Ciudad'], prefix='Ciudad')
print(data.head())

# Selecciona las características que quieres utilizar para la regresión
X = data[['Mes', 'Datos vendidos', 'Ciudad_Medellín', 'Ciudad_Cali', 'Ciudad_Bogotá', 'Ciudad_Barranquilla']]
y = data['Cantidad clientes']

# Utiliza un modelo de regresión polinómica
degree = 2  # Puedes ajustar el grado del polinomio según tus necesidades
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X, y)

# Realiza predicciones
predicciones = polyreg.predict(X)

plt.scatter(y, predicciones, c='fuchsia')
plt.xlabel("Valores Reales")
plt.ylabel("Predicciones")
plt.title("Regresión Polinómica (Grado {})".format(degree))
plt.show()
