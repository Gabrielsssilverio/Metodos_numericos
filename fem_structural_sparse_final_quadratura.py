# -*- coding: utf-8 -*-

import array
import meshio
import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse, linalg
from scipy.spatial import Delaunay

plt.style.use('dark_background')

'''
==============================================
FEM2DSTRUCT
==============================================
'''
class fem2Dstruct(): # modified
    def __init__(self):
        self.nnodes = 0
        self.nelements = 0
        
        self.bc_type = []
        self.bc_nodes = []
        self.bc_values = []
        return

    def create_nodes(self, coords):
        self.nodes = coords
        self.nnodes = len(self.nodes)
        return

    def create_elements(self, connectivities):
        self.elements = []
        self.connectivities = connectivities
        for item in connectivities:
            if len(item) == 3:
                element = ST3(item)  # modified
            if len(item) == 4:
                element = SQ4(item)

            self.elements.append(element)
        self.nelements = len(self.elements)
        return

    def define_props(self, props):
        for i, element in enumerate(self.elements):
            element.props = props
        return

    def create_bc(self, bc_type, nodes, values):
        self.bc_type.append(bc_type)
        self.bc_nodes.append(nodes)
        self.bc_values.append(values)
        return

    def solve(self):

        rows = []
        cols = []
        values = []

        for element in self.elements:
            r, c, v = element.Kmatrix(self.nodes)
            rows.append(r)
            cols.append(c)
            values.append(v)

        rows = np.array(rows, dtype='int').flatten()
        cols = np.array(cols, dtype='int').flatten()
        values = np.array(values, dtype='float').flatten()

        Kglobal = sparse.csr_matrix((values, (rows, cols)), shape=((2*self.nnodes, 2*self.nnodes)))

        self.Kglobal = Kglobal
        
        # 1) Montagem do vetor fglobal
        fglobal = np.zeros(2 * self.nnodes)
        
        w = 1e30
        for bc_type, bc_node, bc_value in zip(self.bc_type, self.bc_nodes, self.bc_values):
            if bc_type == 'ux':
                ngdl = 2 * bc_node
                Kglobal[ngdl, ngdl] += w
                fglobal[ngdl] = bc_value * w
            elif bc_type == 'uy':
                ngdl = 2 * bc_node + 1
                Kglobal[ngdl, ngdl] += w
                fglobal[ngdl] = bc_value * w   
            else:
                ngdl = 2 * bc_node
                fglobal[ngdl] += bc_value[0]
                fglobal[ngdl+1] += bc_value[1]
                

        self.Kglobal = Kglobal
        self.fglobal = fglobal
        
        # 4) Resolver o problema
        self.u = spsolve(Kglobal, fglobal)
        return

    def plot(self):
        plt.figure()
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.connectivities, '-w', linewidth=0.5)
        plt.axis('off')
        plt.axis('equal')
        
        plt.figure()
        plt.tripcolor(self.nodes[:,0], self.nodes[:,1], self.connectivities, self.u[0::2], shading='gouraud')
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.connectivities, '-w', linewidth=0.5)
        plt.colorbar()
        plt.axis('off')
        plt.axis('equal')
        plt.title('Displacement - ux')


        ux_scale = self.u[0::2] * 0.1 * np.max(self.nodes) / np.max(abs(self.u))
        uy_scale = self.u[1::2] * 0.1 * np.max(self.nodes) / np.max(abs(self.u))
        plt.figure()
        plt.tripcolor(self.nodes[:,0], self.nodes[:,1], self.connectivities, self.u[1::2], shading='gouraud')
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.connectivities, '-w', linewidth=0.5)
        plt.triplot(self.nodes[:,0]+ux_scale, self.nodes[:,1]+uy_scale, self.connectivities, '-w', linewidth=0.5)
        plt.colorbar()
        plt.axis('off')
        plt.axis('equal')
        plt.title('Displacement - uy')

        plt.show()

class ST3():  # modified
    def __init__(self, nodes):
        self.nodes = nodes
        self.props = 0.0
        return

    def Kmatrix(self, coords):
        x = coords[self.nodes, 0]
        y = coords[self.nodes, 1]

        B = np.zeros((3,6))  # modified
        B[0][0] = y[1] - y[2] # modified
        B[0][2] = y[2] - y[0] # modified
        B[0][4] = y[0] - y[1] # modified

        B[1][1] = x[2] - x[1] # modified
        B[1][3] = x[0] - x[2] # modified
        B[1][5] = x[1] - x[0] # modified
        
        B[2][0] = B[1][1] # modified
        B[2][1] = B[0][0] # modified
        B[2][2] = B[1][3] # modified
        B[2][3] = B[0][2] # modified
        B[2][4] = B[1][5] # modified
        B[2][5] = B[0][4] # modified
        
        A = 0.5*(x[0]*y[1] + y[0]*x[2] + x[1]*y[2] - x[2]*y[1] - x[0]*y[2] - x[1]*y[0])

        B = (1.0/(2*A)) * B
        
        E = self.props['Young'] # added
        nu = self.props['Poisson'] # added
        
        C = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1.0 - nu)/2]])

        K = B.T @ C @ B * A

        self.area = A
        self.centroid = [np.mean(x), np.mean(y)]
        
        nodes_list = np.array([2*self.nodes[0], 2*self.nodes[0]+1, \
                               2*self.nodes[1], 2*self.nodes[1]+1, \
                               2*self.nodes[2], 2*self.nodes[2]+1])
        
        [cols, rows] = np.meshgrid(nodes_list, nodes_list)
        ind_rows = rows.flatten()
        ind_cols = cols.flatten()
        values = K.flatten()

        return ind_rows, ind_cols, values
    
class SQ4():  # modified
    def __init__(self, nodes):
        self.nodes = nodes
        self.props = 0.0
        return

    def Kmatrix(self, coords):
        x = coords[self.nodes, 0]
        y = coords[self.nodes, 1]

        N1 = lambda xi, eta: (1/4) * (1 - xi) * (1 - eta)
        N2 = lambda xi, eta: (1/4) * (1 + xi) * (1 - eta)
        N3 = lambda xi, eta: (1/4) * (1 + xi) * (1 + eta)
        N4 = lambda xi, eta: (1/4) * (1 - xi) * (1 + eta)

        dN1dxi = lambda xi, eta: (1/4) * -(1 - eta)
        dN2dxi = lambda xi, eta: (1/4) * +(1 - eta)
        dN3dxi = lambda xi, eta: (1/4) * +(1 + eta)
        dN4dxi = lambda xi, eta: (1/4) * -(1 + eta)

        dN1deta = lambda xi, eta: (1/4) * -(1 - xi) 
        dN2deta = lambda xi, eta: (1/4) * -(1 + xi)
        dN3deta = lambda xi, eta: (1/4) * +(1 + xi) 
        dN4deta = lambda xi, eta: (1/4) * +(1 - xi)

        N = lambda xi, eta: np.array([N1(xi, eta), N2(xi, eta), N3(xi, eta), N4(xi, eta)])
        dNdxi = lambda xi, eta: np.array([dN1dxi(xi, eta), \
                                        dN2dxi(xi, eta), \
                                        dN3dxi(xi, eta), \
                                        dN4dxi(xi, eta)])
        dNdeta = lambda xi, eta: np.array([dN1deta(xi, eta), \
                                        dN2deta(xi, eta), \
                                        dN3deta(xi, eta), \
                                        dN4deta(xi, eta)])

        abscissa, weight = np.polynomial.legendre.leggauss(2)

        K = np.zeros((8,8))

        E = self.props['Young'] # added
        nu = self.props['Poisson'] # added
        C = (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1.0 - nu)/2]])

        for xi, w_xi in zip(abscissa, weight):
            for eta, w_eta in zip(abscissa, weight):           
                
                xp = N(xi, eta) @ x
                yp = N(xi, eta) @ y
                
                J11 = dNdxi(xi, eta) @ x
                J12 = dNdeta(xi, eta) @ x
                J21 = dNdxi(xi, eta) @ y
                J22 = dNdeta(xi, eta) @ y
                
                J = np.array([[J11, J12], [J21, J22]])

                B = np.zeros((3,8))
                B[0][0] = dN1dxi(xi, eta)
                B[0][2] = dN2dxi(xi, eta)
                B[0][4] = dN3dxi(xi, eta)
                B[0][6] = dN4dxi(xi, eta)

                B[1][1] = dN1deta(xi, eta)
                B[1][3] = dN2deta(xi, eta)
                B[1][5] = dN3deta(xi, eta)
                B[1][7] = dN4deta(xi, eta)

                B[2][0] = dN1deta(xi, eta)
                B[2][1] = dN1dxi(xi, eta)
                B[2][2] = dN2deta(xi, eta)
                B[2][3] = dN2dxi(xi, eta)
                B[2][4] = dN3deta(xi, eta)
                B[2][5] = dN3dxi(xi, eta)
                B[2][6] = dN4deta(xi, eta)
                B[2][7] = dN4dxi(xi, eta)

                aux = w_xi * w_eta * B.T @ C @ B * np.abs(np.linalg.det(J))
                
                K += aux

        nodes_list = np.array([2*self.nodes[0], 2*self.nodes[0]+1, \
                               2*self.nodes[1], 2*self.nodes[1]+1, \
                               2*self.nodes[2], 2*self.nodes[2]+1,
                               2*self.nodes[3], 2*self.nodes[3]+1])
        
        [cols, rows] = np.meshgrid(nodes_list, nodes_list)
        ind_rows = rows.flatten()
        ind_cols = cols.flatten()
        values = K.flatten()

        return ind_rows, ind_cols, values

'''
===============================================
MAIN
===============================================
'''
plt.close('all')

problem = fem2Dstruct()

coords = np.array([[0.0, 0.0],
                   [0.5, 0.0],
                   [1.0, 0.0],
                   [0.0, 0.1],
                   [0.5, 0.1],
                   [1.0, 0.1]
                   ])

connectivities = [[0, 1, 4 ,3], [1, 2, 5, 4]]

# (1) - Geometry and mesh
problem.create_nodes(coords)
problem.create_elements(connectivities)

# (2) - Properties
props = {'Young': 72e9, 'Poisson':0.3}
problem.define_props(props)

# (3) - Boundary conditions
problem.create_bc('ux', 0, 0.0)
problem.create_bc('ux', 3, 0.0)
problem.create_bc('uy', 0, 0.0)
problem.create_bc('F', 2, [0.0, -1000])

# (4) - Solve
problem.solve()

# Numerical
uymin = problem.u
print(f"Numerical: {uymin} \n")

'''
# (5) - Postprocessing
#problem.plot()

# Analytical
delta = -P * L**3 / (3 * props['Young'] * W**3 / 12)

print(f"Analytical: {delta} \n")
'''
