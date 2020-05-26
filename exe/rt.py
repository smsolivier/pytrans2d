#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 

Ne = 10 
p = 2
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])
mesh = RectMesh(Ne, Ne)
rt = RTSpace(mesh, LobattoBasis, LegendreBasis, p)
l2 = L2Space(mesh, LegendreBasis, p)
print('rt Nu = {}'.format(rt.Nu))
print('l2 Nu = {}'.format(l2.Nu))

M = Assemble(rt, VectorFEMassIntegrator, lambda x: -1, 2*(p+2)+1)
D = MixAssemble(l2, rt, VectorFEDivIntegrator, 1, 2*(p+1)+1)
Q = lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
# Q = lambda x: 1 
f = AssembleRHS(l2, DomainIntegrator, Q, 2*(p+1)+1)

A = sp.bmat([[M, D.transpose()], [D, None]]).tocsc()
rhs = np.concatenate((np.zeros(rt.Nu), f))

x = spla.spsolve(A, rhs) 

T = GridFunction(l2)
T.data = x[rt.Nu:]
q = GridFunction(rt)
q.data = x[:rt.Nu]

ip, w = quadrature.Get1D(2*(p+1)+1)
jump = 0
for face in mesh.iface:
	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n]) 
		xi2 = face.ipt2.Transform(ip[n]) 
		vp = rt.el.CalcPhysVShape(face.trans1, xi1)
		vm = rt.el.CalcPhysVShape(face.trans2, xi2) 
		nor = face.face.Normal(ip[n]) 
		pdofs = q.data[rt.dofs[face.ElNo1]]
		mdofs = q.data[rt.dofs[face.ElNo2]]
		diff = np.dot(nor, np.dot(vp, pdofs) - np.dot(vm, mdofs))
		jump += (np.dot(diff, diff)) * w[n] * face.face.Jacobian(ip[n]) 

print('jump = {:.3e}'.format(np.sqrt(jump)))

Tex = lambda x: np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
err = T.L2Error(Tex, 2*(p+1)+2)
print('err = {:.3e}'.format(err))

qex = lambda x: [-np.pi*np.cos(np.pi*x[0])*np.sin(np.pi*x[1]), 
	-np.pi*np.sin(np.pi*x[0])*np.cos(np.pi*x[1])]
qerr = q.L2Error(qex, 2*(p+1)+2)
print('qerr = {:.3e}'.format(qerr))

mesh.WriteVTK('solution', cell={'T':T.ElementData(), 'q':q.ElementData()})
