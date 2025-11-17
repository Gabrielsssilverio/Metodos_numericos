# Exercicio 4 - Lista 1 (versão modular com teste de convergência)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator

# ------------------------------
# Parâmetros do problema
# ------------------------------
Lx = 1.0   # comprimento da chapa em x
Ly = 1.0   # comprimento da chapa em y
k = 0.3    # condutividade térmica
s = 0
q = 100    # fluxo de calor (W/m2)
hc = 20    # coeficiente de troca de calor
Tinf = 0   # temperatura do meio

# ------------------------------
# Função para resolver o sistema
# ------------------------------
def solve_heat(Nx, Ny):
    # Discretização
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    [xg, yg] = np.meshgrid(x,y)

    # Regiões da malha
    region_B = np.arange(0,Nx,1)
    region_R = np.arange(Nx-1,Nx*Ny,Nx)
    region_T = np.arange(Nx*(Ny-1),Nx*Ny)
    region_L = np.arange(0,Nx*(Ny-1)+1,Nx)

    # Montagem A e b
    A = np.zeros((Nx*Ny,Nx*Ny))
    b = np.zeros(Nx*Ny)

    for i in range(Nx*Ny):
        # Cantos
        if i == 0:  # canto inferior esquerdo
            A[i,i] = -(2/dx**2 + 2/dy**2)
            A[i,i+1] = 2/dx**2
            A[i,i+Nx] = 2/dy**2
            b[i] = -s/k - (2*q/(dx*k))   

        elif i == Nx-1:  # canto inferior direito
            A[i,i] = -(2/dx**2 + 2/dy**2 + 2*hc/(dx*k))
            A[i,i-1] = 2/dx**2
            A[i,i+Nx] = 2/dy**2
            b[i] = -s/k - 2*hc*Tinf/(dx*k)

        elif i == Nx*(Ny-1):  # canto superior esquerdo
            A[i,i] = -(2/dx**2 + 2/dy**2)
            A[i,i+1] = 2/dx**2
            A[i,i-Nx] = 2/dy**2
            b[i] = -s/k - (2*q/(dx*k))

        elif i == Nx*Ny - 1:  # canto superior direito
            A[i,i] = -(2/dx**2 + 2/dy**2 + 2*hc/(dx*k))
            A[i,i-1] = 2/dx**2
            A[i,i-Nx] = 2/dy**2
            b[i] = -s/k - 2*hc*Tinf/(dx*k)

        # Bordas (sem cantos)
        elif i in region_B:
            A[i,i-1] = 1/dx**2
            A[i,i]   = -(2/dx**2 + 2/dy**2 + 2*hc/(dy*k))
            A[i,i+1] = 1/dx**2
            A[i,i+Nx] = 2/dy**2
            b[i] = -s/k - 2*hc*Tinf/(dy*k)

        elif i in region_T:
            A[i,i-1] = 1/dx**2
            A[i,i]   = -(2/dx**2 + 2/dy**2 + 2*hc/(dy*k))
            A[i,i+1] = 1/dx**2
            A[i,i-Nx] = 2/dy**2
            b[i] = -s/k - 2*hc*Tinf/(dy*k)

        elif i in region_L:
            A[i,i-Nx] = 1/dy**2
            A[i,i+1]  = 2/dx**2
            A[i,i]    = -(2/dx**2 + 2/dy**2)
            A[i,i+Nx] = 1/dy**2
            b[i] = -s/k - (2*q/(dx*k))

        elif i in region_R:
            A[i,i-Nx] = 1/dy**2
            A[i,i-1]  = 2/dx**2
            A[i,i]    = -(2/dx**2 + 2/dy**2 + 2*hc/(dx*k))
            A[i,i+Nx] = 1/dy**2
            b[i] = -s/k - 2*hc*Tinf/(dx*k)

        # Pontos internos
        else:
            A[i,i-Nx] = 1/dy**2
            A[i,i-1]  = 1/dx**2
            A[i,i]    = -(2/dx**2 + 2/dy**2)
            A[i,i+1]  = 1/dx**2
            A[i,i+Nx] = 1/dy**2
            b[i] = -s/k

    # Resolvendo sistema
    T = np.linalg.solve(A, b)
    return T.reshape((Ny,Nx)), x, y, xg, yg

# Função para obter temperatura no ponto central da borda inferior
def get_bottom_center_temp(Tmatrix, Nx):
    return Tmatrix[0, Nx//2]
# ------------------------------
# Teste de convergência
# ------------------------------
vetor_medicao = []
vetor_diferenca = [0]
eixo_x = []

tol = 1e-4
N_min = 5
N_max = 85
diferenca = np.inf
best_N = None
Tc_ant = None  # valor do passo anterior

for Nx in range(N_min, N_max+1, 2):
    Ny = Nx
    Tmatrix, x, y, xg, yg = solve_heat(Nx, Ny)
    valor = get_bottom_center_temp(Tmatrix, Nx)
    vetor_medicao.append(valor)
    eixo_x.append(Nx)
    if Tc_ant is not None:
        diferenca = abs(valor - Tc_ant)
        vetor_diferenca.append(diferenca)
    Tc_ant = valor
    print(f"N={Nx}, Tc={valor:.6f}, diferença={diferenca:.2e}")
    if diferenca < tol:
        best_N = Nx
        print(f"\n✅ Convergência atingida com N={best_N}")
        break
if best_N is None:
    best_N = Nx

# Atualiza Tmatrix com malha convergida
Tmatrix, x, y, xg, yg = solve_heat(best_N, best_N)
Tc_final = get_bottom_center_temp(Tmatrix, best_N)
print("Temperatura central inferior final:", Tc_final)

# tol = 1e-4
# Tc_old = None
# best_N = None
# for N in [5,10,20, 30, 40, 50, 60,70, 80]:
#     Tmatrix, x, y, xg, yg = solve_heat(N, N)
#     Tc = get_bottom_center_temp(Tmatrix, N)
#     if Tc_old is not None:
#         erro = abs(Tc - Tc_old)
#         print(f"N={N}, Tc={Tc:.6f}, erro={erro:.2e}")
#         if erro < tol:
#             best_N = N
#             print(f"✅ Convergência atingida com N={N}")
#             break
#     Tc_old = Tc
# ------------------------------
# Gráficos finais com malha convergida
# ------------------------------
if best_N is None:
    best_N = N   # se não convergiu dentro do loop

Tmatrix, x, y, xg, yg = solve_heat(best_N, best_N)

# Campo de temperatura (imshow)
plt.figure(figsize=(6,5))
plt.imshow(Tmatrix, origin="lower",
           extent=[0, Lx, 0, Ly],
           cmap="hot", aspect="equal")
plt.colorbar(label="Temperatura [K]")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.title("Campo de Temperatura na Chapa")
plt.show()
