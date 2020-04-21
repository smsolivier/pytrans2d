#!/usr/bin/env python3

from trans2d import * 
import pytest 

mesh = RectMesh(3,3)
space = L2Space(mesh, LegendreBasis, 2)
quad = LevelSym(16)
psi = TVector(space, quad) 
qdf = QDFactors(space, quad)

def test_isotropic():
	psi.Project(lambda x, Omega: 1)
	qdf.Compute(psi) 

	trans = mesh.trans[0]
	E = qdf.EvalTensor(trans, [0,0])
	assert(E==pytest.approx(np.eye(2)/3))

	fi = mesh.iface[0]
	G = qdf.EvalG(fi, 0)
	assert(G==pytest.approx(.5, abs=1e-2))
