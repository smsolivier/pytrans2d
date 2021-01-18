import numpy as np 
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 
import time 
import warnings 

from . import sn 
from .. import fem 
from ..ext import linalg 
from .. import utils 

def JumpJumpIntegrator(el1, el2, face, c, qorder):
	elmat = np.zeros((2*el1.Nn, 2*el2.Nn))
	ip, w = fem.quadrature.Get1D(qorder)

	j = np.zeros(el1.Nn + el2.Nn)
	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n])
		xi2 = face.ipt2.Transform(ip[n])

		s1 = el1.CalcShape(xi1)
		s2 = el2.CalcShape(xi2)
		j[:el1.Nn] = s1 
		j[el1.Nn:] = -s2 

		linalg.AddOuter(face.face.Jacobian(ip[n])*c*w[n], j, j, elmat) 

	return elmat 

def MixJumpVJumpIntegrator(el1, el2, face, c, qorder):
	elmat = np.zeros((el1[0].Nn+el1[1].Nn, 2*el2[0].Nn+2*el2[1].Nn))
	ip, w = fem.quadrature.Get1D(qorder)

	j1 = np.zeros(el1[0].Nn+el1[1].Nn)
	j2 = np.zeros(2*el2[0].Nn+2*el2[1].Nn)
	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n])
		xi2 = face.ipt2.Transform(ip[n])
		nor = face.face.Normal(ip[n])

		s1 = el1[0].CalcShape(xi1)
		s2 = el1[1].CalcShape(xi2)
		vs1 = el2[0].CalcVShape(xi1)
		vs2 = el2[1].CalcVShape(xi2)

		j1[:el1[0].Nn] = s1
		j1[el1[0].Nn:] = -s1 
		j2[:2*el2[0].Nn] = np.dot(nor, vs1)
		j2[2*el2[0].Nn:] = -np.dot(nor, vs2)

		linalg.AddOuter(face.face.Jacobian(ip[n])*c*w[n], j1, j2, elmat)

	return elmat 

def MixJumpVAvgIntegrator(el1, el2, face, c, qorder):
	elmat = np.zeros((2*el1[0].Nn, 4*el2[0].Nn))
	ip, w = fem.quadrature.Get1D(qorder)

	j = np.zeros(2*el1[0].Nn)
	a = np.zeros(4*el2[0].Nn)
	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n])
		xi2 = face.ipt2.Transform(ip[n])
		nor = face.face.Normal(ip[n])

		s1 = el1[0].CalcShape(xi1)
		s2 = el1[1].CalcShape(xi2)
		vs1 = el2[0].CalcVShape(xi1)
		vs2 = el2[1].CalcVShape(xi2)
		j[:el1[0].Nn] = s1 
		j[el1[0].Nn:] = -s2
		a[:2*el2[0].Nn] = np.dot(nor, vs1)
		a[2*el2[0].Nn:] = np.dot(nor, vs2)

		# bfac = .5
		bfac = 1 if face.boundary else .5 
		linalg.AddOuter(bfac*face.face.Jacobian(ip[n])*c*w[n], j, a, elmat)

	return elmat 

def VectorJumpJumpIntegrator(el1, el2, face, c, qorder):
	elmat = np.zeros((4*el1.Nn, 4*el1.Nn))
	ip, w = fem.quadrature.Get1D(qorder)

	j = np.zeros(4*el1.Nn)
	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n])
		xi2 = face.ipt2.Transform(ip[n])
		nor = face.face.Normal(ip[n])

		vs1 = el1.CalcVShape(xi1)
		vs2 = el2.CalcVShape(xi2)
		j[:2*el1.Nn] = np.dot(nor, vs1)
		j[2*el1.Nn:] = -np.dot(nor, vs2)

		linalg.AddOuter(face.face.Jacobian(ip[n])*w[n]*c, j, j, elmat)

	return elmat 

def VectorJumpAvgIntegrator(el1, el2, face, c, qorder):
	elmat = np.zeros((4*el1[0].Nn, 2*el2[0].Nn))
	ip, w = fem.quadrature.Get1D(qorder)

	j = np.zeros(4*el1[0].Nn)
	a = np.zeros(2*el2[0].Nn)
	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n])
		xi2 = face.ipt2.Transform(ip[n])
		nor = face.face.Normal(ip[n])

		vs1 = el1[0].CalcVShape(xi1)
		vs2 = el1[1].CalcVShape(xi2)
		s1 = el2[0].CalcShape(xi1)
		s2 = el2[1].CalcShape(xi2)

		j[:2*el1[0].Nn] = np.dot(nor, vs1)
		j[2*el1[0].Nn:] = -np.dot(nor, vs2)
		a[:el2[0].Nn] = s1 
		a[el2[0].Nn:] = s2 

		# not sure why this needs to be .5 on bdr faces 
		bfac = .5
		linalg.AddOuter(bfac*face.face.Jacobian(ip[n])*w[n]*c, j, a, elmat)

	return elmat 

class P1SA(sn.Sn):
	def __init__(self, sweep):
		sn.Sn.__init__(self, sweep) 
		self.J_space = fem.L2Space(self.space.mesh, type(self.space.el.basis), self.p, 2) 

		self.Mt = 3*fem.Assemble(self.J_space, fem.VectorMassIntegrator, sweep.sigma_t, 2*self.p+1)
		self.sigma_a = lambda x: self.sweep.sigma_t(x) - self.sweep.sigma_s(x)
		self.Ma = fem.Assemble(self.space, fem.MassIntegrator, self.sigma_a, 2*self.p+1)
		self.D = fem.MixAssemble(self.space, self.J_space, fem.WeakMixDivIntegrator, 1, 2*self.p+1)
		GT = fem.MixAssemble(self.space, self.J_space, fem.MixDivIntegrator, 1, 2*self.p+1)
		self.G = -GT.transpose()

		self.Mt += fem.FaceAssembleAll(self.J_space, VectorJumpJumpIntegrator, 1, 2*self.p+1)
		self.G += fem.MixFaceAssembleAll(self.J_space, self.space, VectorJumpAvgIntegrator, 1, 2*self.p+1)
		self.Ma += fem.FaceAssembleAll(self.space, JumpJumpIntegrator, 1/4, 2*self.p+1)
		self.D += fem.MixFaceAssembleAll(self.space, self.J_space, MixJumpVAvgIntegrator, 1, 2*self.p+1)

		A = sp.bmat([[self.Mt, self.G], [self.D, self.Ma]])
		self.lu = spla.splu(A.tocsc()) 
		self.rhs = np.zeros(A.shape[0])

	def SourceIteration(self, psi, niter=50, tol=1e-10):
		phi = self.ComputeScalarFlux(psi)
		phi_old = fem.GridFunction(self.space)
		diff = fem.GridFunction(self.space)
		for n in range(niter):
			start = time.time()
			phi_old.data = phi.data.copy()
			self.sweep.Sweep(psi, phi) 
			phi = self.ComputeScalarFlux(psi)
			diff.data = phi.data - phi_old.data 
			scat = self.sweep.FormScattering(diff).data*4*np.pi
			self.rhs[self.J_space.Nu:] = scat 
			x = self.lu.solve(self.rhs)
			phi.data += x[self.J_space.Nu:]
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.LOUD):
				el = time.time() - start 
				print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(n+1, norm, el))

			if (norm < tol):
				break 

		if (norm > tol):
			warnings.warn('source iteration not converged. final tol = {:.3e}'.format(norm), utils.ToleranceWarning)

		return phi		