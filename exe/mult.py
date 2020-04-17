#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time
import sys 

from trans2d import * 

Ne = 1000 
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
trans = AffineTrans(np.array([[0,0], [1,0], [1,1], [0,1]]))
el = Element(LagrangeBasis, 1)

start = time.time()
for e in range(Ne):
	# elmat = DiffusionIntegrator(el, trans, lambda x: 1, 0)
	elmat = MassIntegrator(el, trans, lambda x: 1, 2)

print('mult = {:.3f} s'.format(time.time() - start))