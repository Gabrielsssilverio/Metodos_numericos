# -*- coding: utf-8 -*-

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
            element = ST3(item)  # modified
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
        #Kglobal = Kglobal + Kglobal.T - sparse.diags(Kglobal.diagonal(), dtype='float')

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


'''
===============================================
MAIN
===============================================
'''
plt.close('all')

problem = fem2Dstruct()

L = 1
W = 0.05
P = -1000

x = np.linspace(0, L, 300)
y = np.linspace(0, W, 10)

[xg, yg] = np.meshgrid(x, y)

x = xg.flatten().reshape(-1, 1)
y = yg.flatten().reshape(-1, 1)

coords = np.concatenate((x, y), axis = 1)

connectivities = Delaunay(coords).simplices

set_left = [k for k, x in enumerate(x) if (x > -1e-8) and (x < 1e-8)]
set_right = [k for k, x in enumerate(x) if (x > -1e-8 + L) and (x < 1e-8 + L)]


# (1) - Geometry and mesh
problem.create_nodes(coords)
problem.create_elements(connectivities)

# (2) - Properties
props = {'Young': 72e9, 'Poisson':0.3}
problem.define_props(props)

# (3) - Boundary conditions

for node in set_left:
    problem.create_bc('ux', node, 0.0)
    problem.create_bc('uy', node, 0.0)

nnodes_right = len(set_right)
for node in set_right:
    problem.create_bc('F', node, [0.0, P / nnodes_right])
    

# (4) - Solve
problem.solve()

# (5) - Postprocessing
problem.plot()

# Analytical
delta = P * L**3 / (3 * props['Young'] * W**3 / 12)

# Numerical
uymin = min(problem.u[1::2])

print(f"Analytical: {delta} \n")
print(f"Numerical: {uymin} \n")
