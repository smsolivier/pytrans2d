#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from OutputCycler import * 
oc = OutputCycler()

Ne = oc.GetOpt(0, 10)
p = oc.GetOpt(1, 0)
if (len(sys.argv)>1):
	Ne = int(sys.argv[1])
if (len(sys.argv)>2):
	p = int(sys.argv[2])
mesh = RectMesh(Ne, Ne)
# h = 1/Ne 
# for i in range(mesh.nodes.shape[0]):
# 	for j in range(2):
# 		if (mesh.nodes[i,j]>0 and mesh.nodes[i,j]<1):
# 			mesh.nodes[i,j] = np.random.normal(mesh.nodes[i,j], h*.1)

bc = 0
theta = np.pi/6
# theta = 0
rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta),np.cos(theta)]])
mesh.nodes = mesh.nodes@rot.T 
mesh = AbstractMesh(mesh.nodes, mesh.ele, 1)
# mesh.Plot()
# plt.show()

rt = RTSpace(mesh, LobattoBasis, LegendreBasis, p)
l2 = L2Space(mesh, LegendreBasis, p)
print('rt Nu = {}'.format(rt.Nu))
print('l2 Nu = {}'.format(l2.Nu))

def RTBdrIntegrator(el, face, c, qorder):
	elvec = np.zeros(el.Nn)
	ip, w = quadrature.Get1D(qorder)

	# for n in range(len(w)):
	# 	nor = face.face.Normal(ip[n])
	# 	xi1 = face.ipt1.Transform(ip[n])
	# 	nhat = face.trans1.F(xi1).T@nor 
	# 	nhat /= np.linalg.norm(nhat) 
	# 	vs = el.CalcVShape(xi1)
	# 	coef = c(face.trans1.Transform(xi1))

	# 	vsn = nhat@coef@vs
	# 	elvec -= vsn * w[n] 

	for n in range(len(w)):
		nor = face.face.Normal(ip[n])
		tau = face.face.Tangent(ip[n])
		xi1 = face.ipt1.Transform(ip[n])
		vs = el.CalcVShape(xi1)
		coef = c(face.trans1.Transform(xi1))
		F = face.trans1.F(xi1)
		J = np.linalg.det(F) 
		Finv = np.linalg.inv(F)

		cn = coef@nor 
		alpha = cn@nor 
		beta = cn@tau 

		contra = 1/J*vs.T@F.T@nor 
		cov = vs.T@Finv@tau 
		elvec -= (alpha*contra + beta*cov) * w[n] * face.face.Jacobian(ip[n])

	# for n in range(len(w)):
	# 	nor = face.face.Normal(ip[n]) 
	# 	xi1 = face.ipt1.Transform(ip[n]) 
	# 	vs = el.CalcPhysVShape(face.trans1, xi1)
	# 	coef = c(face.trans1.Transform(xi1))
	# 	vsn = nor@coef@vs 
	# 	elvec -= vsn * w[n] * face.face.Jacobian(ip[n])

	# for n in range(len(w)):
	# 	nor = face.face.Normal(ip[n]) 
	# 	tan = face.face.Tangent(ip[n])
	# 	xi1 = face.ipt1.Transform(ip[n]) 
	# 	vs = el.CalcVShape(xi1)
	# 	coef = c(face.trans1.Transform(xi1))@nor
	# 	alpha = coef@nor 
	# 	beta = coef@tan 
	# 	F = face.trans1.F(xi1)
	# 	Finv = np.linalg.inv(F) 
	# 	J = np.linalg.det(F) 
	# 	Jg = face.face.Jacobian(ip[n])
	# 	elvec -= w[n] * (alpha*vs.T@(F.T@nor)*Jg/J + beta*vs.T@(Finv@tan)*Jg)

	return elvec 

def VectorFEJumpAvg(el1, el2, face, c, qorder):
	elmat = np.zeros((2*el1[0].Nn, 2*el2[0].Nn))
	ip, w = quadrature.Get1D(qorder) 

	# for n in range(len(w)):
	# 	nor = face.face.Normal(ip[n]) 
	# 	xi1 = face.ipt1.Transform(ip[n]) 
	# 	xi2 = face.ipt2.Transform(ip[n]) 
	# 	nhat = face.trans1.F(xi1).T@nor 
	# 	nhat /= np.linalg.norm(nhat) 
	# 	vs1 = el1[0].CalcVShape(xi1) 
	# 	vs2 = el1[1].CalcVShape(xi2) 
	# 	coef = c(face.trans1.Transform(xi1))
	# 	s1 = el2[0].CalcShape(xi1)
	# 	s2 = el2[1].CalcShape(xi2) 

	# 	F1 = face.trans1.F(xi1)
	# 	F2 = face.trans2.F(xi2)
	# 	J1 = face.trans1.Jacobian(xi1)
	# 	J2 = face.trans2.Jacobian(xi2)
	# 	j = np.concatenate((coef@nhat@vs1, -coef@nhat@vs2))
	# 	a = np.concatenate((s1, s2)) * (1 if face.boundary else .5) 

	# 	elmat += np.outer(j,a) * w[n] 

	for n in range(len(w)):
		nor = face.face.Normal(ip[n])
		tau = face.face.Tangent(ip[n])
		xi1 = face.ipt1.Transform(ip[n])
		xi2 = face.ipt2.Transform(ip[n])
		vs1 = el1[0].CalcVShape(xi1)
		vs2 = el1[1].CalcVShape(xi2)
		coef = c(face.trans1.Transform(xi1))
		s1 = el2[0].CalcShape(xi1)
		s2 = el2[1].CalcShape(xi2)

		cn = coef@nor
		beta = cn@tau
		F1 = face.trans1.F(xi1)
		J1 = np.linalg.det(F1)
		F2 = face.trans2.F(xi2)
		J2 = np.linalg.det(F2)
		Fi1 = np.linalg.inv(F1)
		Fi2 = np.linalg.inv(F2)
		Jf = face.face.Jacobian(ip[n])
		cov1 = vs1.T@Fi1@tau
		cov2 = vs2.T@Fi2@tau

		j = np.concatenate((beta*cov1, -beta*cov2))
		a = np.concatenate((s1, s2)) * (1 if face.boundary else .5) 
		elmat += np.outer(j,a) * w[n] * Jf

	return elmat 

M = Assemble(rt, VectorFEMassIntegrator, lambda x: 1, 2*(p+2)+1)
Ml = Assemble(rt, VectorFEMassIntegratorRowSum, lambda x: 1, 2*(p+2)+1)
Ma = Assemble(l2, MassIntegrator, lambda x: 0, 2*(p+1)+1)
diag = Ml.diagonal() 
Mlinv = sp.diags(1/diag)
alpha = 1/3
beta = 1/10
# beta = 0
def R(x):
	x = rot.T@x
	# offt = beta*np.sin(2*np.pi*x[0])*np.sin(2*np.pi*x[1]) 
	# offb = 15*(1 + np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
	# return np.array([[1./3, offt/offb], [offt/offb, 1./3]])
	return np.array([[alpha, beta], [beta, alpha]])
# R = lambda x: np.eye(2)
D = MixAssemble(l2, rt, VectorFEDivIntegrator, 1, 2*(p+1)+1)
G = MixAssemble(rt, l2, VectorFEDivIntegrator2, R, 2*(p+1)+1)
F = MixFaceAssemble(rt, l2, VectorFEJumpAvg, R, 2*(p+2)+1)
def Q(x):
	x = rot.T@x 
	# return np.pi**2*(beta*-8/15*np.cos(2*np.pi*x[0])*np.cos(2*np.pi*x[1]) + 2/3*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
	return 2*np.pi**2*(-beta*np.cos(np.pi*x[0])*np.cos(np.pi*x[1]) + alpha*np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
# Q = lambda x: 2*np.pi**2*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
f = AssembleRHS(l2, DomainIntegrator, Q, 2*(p+1)+1) 
g = FaceAssembleRHS(rt, RTBdrIntegrator, lambda x: R(x)*bc, 2*(p+2)+1)

# D2 = -D.T 
D2 = G+F
A = sp.bmat([[M, D2], [D, None]]).tocsc()
rhs = np.concatenate((g, f))
# print('nnz(D) =', D.getnnz())
# print('nnz(G) =', D2.getnnz())

x = spla.spsolve(A, rhs) 
# tri = BlockTri(1e-10, 100, 3, False)
# x = tri.Solve(M, Mlinv, D2, D, Ma, rhs) 
# print('it = {}, norm = {:.3e}'.format(tri.it, tri.norm))

T = GridFunction(l2)
T.data = x[rt.Nu:]
q = GridFunction(rt)
q.data = x[:rt.Nu]

def Tex(x):
	x = rot.T@x
	return np.sin(np.pi*x[0])*np.sin(np.pi*x[1]) + bc
err = T.L2Error(Tex, 2*(p+1)+2)
print('err = {:.3e}'.format(err))

mesh.WriteVTK('solution', cell={'T':T.ElementData(), 'q':q.ElementData()})
