#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 17:00:28 2025

@author: angelico
"""

import numpy as np

x = np.array([-1, 1, 3, -2])
y = np.array([-2, -3, 2, 1])

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

abscissa, weight = np.polynomial.legendre.leggauss(3)

v = 0.0

for xi, w_xi in zip(abscissa, weight):
    for eta, w_eta in zip(abscissa, weight):           
        
        xp = N(xi, eta) @ x
        yp = N(xi, eta) @ y
        
        J11 = dNdxi(xi, eta) @ x
        J12 = dNdeta(xi, eta) @ x
        J21 = dNdxi(xi, eta) @ y
        J22 = dNdeta(xi, eta) @ y
        
        J = np.array([[J11, J12], [J21, J22]])
        
        v1 = w_xi * w_eta * (xp**2 + yp**2) * np.abs(np.linalg.det(J))
        
        v += v1

        print(xp, yp, v1)
        print(J)
        print('----------')

print(v)



