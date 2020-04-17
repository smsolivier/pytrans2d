#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

p = 1
x = np.linspace(-1,1,p+1)
X,Y = np.meshgrid(x,x)

N = (p+1)**2
V = np.polynomial.polynomial.polyvander2d(X.flatten(), Y.flatten(), [p,p])
c = np.zeros((N, N))
for i in range(N):
	b = np.zeros(N)
	b[i] = 1 
	c[:,i] = np.linalg.solve(V, b)

print(np.dot(V, c))