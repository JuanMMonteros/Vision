import numpy as np

# Crear la matriz dada
matriz = np.array([
    [2, 2, 5, 6],
    [0, 3, 7, 4],
    [8, 8, 5, 2],
    [1, 5, 6, 1]
])
print("Matriz original:")
print(matriz)

# 1. Seleccionar el subarray [8, 8, 5, 2] (correcto)
subarray = np.array([matriz[2, 0], matriz[2, 1], matriz[2, 2], matriz[2, 3]])
print("\nSubarray [8, 8, 5, 2]:")
print(subarray)

# 2. Poner la diagonal de la matriz en cero (corregido)
matriz_con_diagonal_cero = matriz.copy()  # Creamos una copia para no modificar la original
for i in range(len(matriz_con_diagonal_cero)):
    matriz_con_diagonal_cero[i, i] = 0
print("\nMatriz con diagonal en cero:")
print(matriz_con_diagonal_cero)

# 3. Sumar todos los elementos del array (correcto)
suma_total = np.sum(matriz)
print("\nSuma de todos los elementos:", suma_total)

# 4. Setear valores pares en 0 e impares en 1 (corregido)
matriz_modificada = np.ones_like(matriz_con_diagonal_cero)  # Iniciamos con 1s (impares)
for i in range(matriz.shape[0]):
    for j in range(matriz.shape[1]):
        if matriz[i, j] % 2 == 0:  # Si es par
            matriz_modificada[i, j] = 0

print("\nMatriz con pares=0 e impares=1:")
print(matriz_modificada)
