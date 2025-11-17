# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 22:46:06 2025

@author: PedroUSP
"""

# Example 4 - FDM - 2D - Steady - versão com cantos tratados

import numpy as np
import matplotlib.pyplot as plt

# Input data
Lx = 1
Ly = 1
k = 0.3
h = 20.0
Tinf = 0
s = 0
q = 100.0

# Discretization
Nx = 99  # Numero de pontos em X
Ny = 99  # Numero de pontos em Y
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
dx = x[1] - x[0]
dy = y[1] - y[0]
[xg, yg] = np.meshgrid(x, y)
x = xg.flatten()
y = yg.flatten()

plt.plot(x, y, '.k')
plt.xlabel('x')
plt.ylabel('y')
plt.axis('equal')

#quinas ~ VER A FIGURA DO ESQUEMA (DOC WORD)
pontoA = (Ny - 1) * Nx     
pontoB = 0               
pontoC = Nx - 1     
pontoD = Nx*Ny - 1         
 
region_B = np.arange(0, Nx, 1)                 # bottom
region_R = np.arange(Nx - 1, Nx * Ny, Nx)      # right
region_T = np.arange(Nx * (Ny - 1), Nx * Ny)   # top
region_L = np.arange(0, Nx * (Ny - 1) + 1, Nx) # left

plt.plot(x[region_B], y[region_B], 'sb', label='Bottom')
plt.plot(x[region_R], y[region_R], 'sr', label='Right')
plt.plot(x[region_T], y[region_T], 'sg', label='Top')
plt.plot(x[region_L], y[region_L], 'sm', label='Left')
plt.plot(x, y, '.k')
# plt.legend()
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi, yi, ' ' + str(i))
plt.show()

# Assembly of matrices and vectors
A = np.zeros((Nx * Ny, Nx * Ny))
b = np.zeros(Nx * Ny)

for i in range(Nx * Ny):

    # os pontos das extremidades (quinas)
    if i == pontoA:  # canto a
        A[i, i] = -2/dx**2 - 2/dy**2 - 2*h/(k*dy)
        A[i, i+1] = 2/dx**2
        A[i, i-Nx] = 2/dy**2
        b[i] = -2*q/(k*dx) - 2*h*Tinf/(k*dy)
        
    elif i == pontoB:  # canto b
        A[i, i] = -2/dx**2 - 2/dy**2 - 2*h/(k*dy)
        A[i, i+1] = 2/dx**2
        A[i, i+Nx] = 2/dy**2
        b[i] = -2*q/(k*dx) - 2*h*Tinf/(k*dy)

    elif i == pontoC:  # canto c
        A[i, i] = -2/dx**2 - 2/dy**2 - 2*h/(k*dx) - 2*h/(k*dy)
        A[i, i-1] = 2/dx**2
        A[i, i+Nx] = 2/dy**2
        b[i] = -2*h*Tinf/(k*dx) - 2*h*Tinf/(k*dy)

    elif i == pontoD:  # canto d
        A[i, i] = -2/dx**2 - 2/dy**2 - 2*h/(k*dx) - 2*h/(k*dy)
        A[i, i-1] = 2/dx**2
        A[i, i-Nx] = 2/dy**2
        b[i] = -2*h*Tinf/(k*dx) - 2*h*Tinf/(k*dy)



    # as regiões conforme descrito no documento word
    elif i in region_B:  # bottom
        A[i, i] = -2/dx**2 - 2/dy**2 - 2*h/(k*dy)
        A[i, i+1] = 1/dx**2
        A[i, i-1] = 1/dx**2
        A[i, i+Nx] = 2/dy**2
        b[i] = -2*h*Tinf/(k*dy)

    elif i in region_R:  # right
        A[i, i] = -2/dx**2 - 2/dy**2 - 2*h/(k*dx)
        A[i, i-1] = 2/dx**2
        A[i, i+Nx] = 1/dy**2
        A[i, i-Nx] = 1/dy**2
        b[i] = -2*h*Tinf/(k*dx)

    elif i in region_T:  # top
        A[i, i] = -2/dx**2 - 2/dy**2 - 2*h/(k*dy)
        A[i, i+1] = 1/dx**2
        A[i, i-1] = 1/dx**2
        A[i, i-Nx] = 2/dy**2
        b[i] = -2*h*Tinf/(k*dy)

    elif i in region_L:  # left
        A[i, i] = -2/dx**2 - 2/dy**2
        A[i, i+1] = 2/dx**2
        A[i, i+Nx] = 1/dy**2
        A[i, i-Nx] = 1/dy**2
        b[i] = -2*q/(k*dx)

# nós internos
    else:
        A[i, i] = -2/dx**2 - 2/dy**2
        A[i, i-1] = 1/dx**2
        A[i, i+1] = 1/dx**2
        A[i, i-Nx] = 1/dy**2
        A[i, i+Nx] = 1/dy**2
        b[i] = 0

T = np.linalg.solve(A, b)

# Solução reshape
Tmatrix = T.reshape((Ny, Nx))

# Criar malha de coordenadas físicas
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Plot em coordenadas físicas
plt.figure()
plt.pcolormesh(X, Y, Tmatrix, cmap='hot', shading='auto')
plt.colorbar(label="Temperatura [°C]")
plt.xlabel("Lx [m]")
plt.ylabel("Ly [m]")
plt.title("Campo de Temperatura")
plt.axis("equal")
plt.show()
# Solving

