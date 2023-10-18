import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Datos de tiempo y tamaño de la población
tiempo = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
tamaño_poblacion = np.array([10, 20, 30, 50, 80, 130, 210, 340, 550, 890])

# Definir la función logística para el modelo
def modelo_logistico(t, a, b, c):
    return c / (1 + np.exp(-a * (t - b)))

# Ajustar el modelo a los datos
parametros, covarianza = curve_fit(modelo_logistico, tiempo, tamaño_poblacion)

# Obtener los parámetros ajustados
a, b, c = parametros

# Generar puntos para la curva ajustada
tiempo_predicho = np.linspace(0, 10, 100)
tamaño_predicho = modelo_logistico(tiempo_predicho, a, b, c)

# Graficar los datos y la curva ajustada
plt.scatter(tiempo, tamaño_poblacion, label='Datos reales')
plt.plot(tiempo_predicho, tamaño_predicho, 'r', label='Curva ajustada (Modelo Logístico)')
plt.xlabel('Tiempo')
plt.ylabel('Tamaño de la Población')
plt.legend()
plt.show()

# Imprimir los parámetros ajustados
print(f'Parámetros ajustados: a = {a}, b = {b}, c = {c}')
