#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 

mesh = RectMesh(3,3)

for edge in mesh.graph.es:
	print('{:>2}: {} --> {}'.format(edge.index, edge.source, edge.target))

trans = AffineFaceTrans(np.array([[-1,-1], [1,-1]]))
print(trans.Transform(-1))