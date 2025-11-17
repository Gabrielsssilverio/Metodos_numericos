# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 10:16:28 2019
"""

import meshio
import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse, linalg

plt.style.use('dark_background')

'''
==============================================
FEM2DHT
==============================================
'''
class fem2dht():
    def __init__(self):
        self.nnodes = 0
        self.nelements = 0

    def create_nodes(self, coords):
        self.nodes = coords
        self.nnodes = len(self.nodes)

    def create_elements(self, connectivities):
        self.elements = []
        self.connectivities = connectivities
        for item in connectivities:
            element = HT3(item)
            self.elements.append(element)
        self.nelements = len(self.elements)

    def define_props(self, props):
        for i, element in enumerate(self.elements):
            element.props = props

    def create_bc(self, bc_type, nodes, values):
        self.bcs_nodes = nodes
        self.bcs_values = values

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

        Kglobal = sparse.csr_matrix((values, (rows, cols)), shape=((self.nnodes, self.nnodes)))
        Kglobal = Kglobal + Kglobal.T - sparse.diags(Kglobal.diagonal(), dtype='float')

        self.Kglobal = Kglobal
        # 1) Montagem do vetor fglobal
        fglobal = np.zeros(self.nnodes)

        # 2) Aplicação das BCs no vetor fglobal
        # 3) Aplicar as condições de contorno na matriz Kglobal
        w = 1e20
        nbcs = len(self.bcs_nodes)
        for k in range(nbcs):
            node = self.bcs_nodes[k]
            fglobal[node] = self.bcs_values[k] * w
            Kglobal[node, node] += w

        self.Kglobal = Kglobal
        self.fglobal = fglobal
        # 4) Resolver o problema
        self.T = spsolve(Kglobal, fglobal)

    def plot(self):
        plt.figure()
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.connectivities, '-w', linewidth=0.5)
        plt.axis('off')
        plt.axis('equal')

        plt.figure()
        plt.tripcolor(self.nodes[:,0], self.nodes[:,1], self.connectivities, self.T, shading='gouraud')
        plt.triplot(self.nodes[:,0], self.nodes[:,1], self.connectivities, '-w', linewidth=0.5)
        plt.colorbar()
        plt.axis('off')
        plt.axis('equal')
        plt.title('Temperature')

        plt.show()



class HT3():
    def __init__(self, nodes):
        self.nodes = nodes
        self.props = 0.0

    def Kmatrix(self, coords):
        x = coords[self.nodes, 0]
        y = coords[self.nodes, 1]

        B = np.zeros((2,3))
        B[0][0] = y[1] - y[2]
        B[0][1] = y[2] - y[0]
        B[0][2] = y[0] - y[1]

        B[1][0] = x[2] - x[1]
        B[1][1] = x[0] - x[2]
        B[1][2] = x[1] - x[0]

        A = 0.5*(x[0]*y[1] + y[0]*x[2] + x[1]*y[2] - x[2]*y[1] - x[0]*y[2] - x[1]*y[0])

        B = (1.0/(2*A)) * B

        K = self.props * np.matmul(B.transpose(), B) * A

        self.area = A
        self.centroid = [np.mean(x), np.mean(y)]

        ind_rows = [self.nodes[0], self.nodes[0], self.nodes[0], self.nodes[1], self.nodes[1], self.nodes[2]]
        ind_cols = [self.nodes[0], self.nodes[1], self.nodes[2], self.nodes[1], self.nodes[2], self.nodes[2]]
        values =   [K[0,0], K[0,1], K[0,2], K[1,1], K[1,2], K[2,2]]

        return ind_rows, ind_cols, values


'''
===============================================
MAIN
===============================================
'''
plt.close('all')

problem = fem2dht()

mesh = meshio.read('ex1_mesh2_tri.msh')
coords = np.array(mesh.points[:,0:2])
connectivities = mesh.cells['triangle']

# Implementação do elemento Q4
# connectivities = ([0, 1, 2, 3])

# (1) - Geometry and mesh
problem.create_nodes(coords)
problem.create_elements(connectivities)

# (2) - Properties
problem.define_props(5)

# (3) - Boundary conditions
nodes = [0, 2, 4]
values = [10, 5, 7.5]
problem.create_bc('T', nodes, values)

# (4) - Solve
problem.solve()

# (5) - Postprocessing
problem.plot()
