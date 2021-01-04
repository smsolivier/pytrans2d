#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from OutputCycler import OutputCycler

oc = OutputCycler()

Ne = oc.GetOpt(0, 10)
p = 2
alpha = 1e5
mesh = RectMesh(Ne, Ne)

sfes = L2Space(mesh, LegendreBasis, p, 1)
vfes = L2Space(mesh, LegendreBasis, p, 2) 

Minv = Assemble(vfes, InverseVectorMassIntegrator, lambda x: 1, 2*p+1) 
D = MixAssemble(sfes, vfes, WeakMixDivIntegrator, 1, 2*p+1) 
F = MixFaceAssembleAll(sfes, vfes, MixJumpVAvgIntegrator, 1, 2*p+1) 
B = D+F 
f = AssembleRHS(sfes, DomainIntegrator, lambda x: -2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+1)

alpha = np.geomspace(.1, 1e6, 10)
err = np.zeros(len(alpha))
for i,a in enumerate(alpha):
	A = FaceAssembleAll(sfes, JumpJumpIntegrator, a, 2*p+1)
	S = A - B*Minv*B.transpose()

	phi = GridFunction(sfes)
	phi.data = spla.spsolve(S, f) 

	err[i] = phi.L2Error(lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1]), 2*p+2)
	# mesh.WriteVTK('solution', cell={'phi':phi.ElementData()})

plt.loglog(alpha, err, '-o')
plt.xlabel('Penalty Parameter')
plt.ylabel('LDG Error')
plt.show()
