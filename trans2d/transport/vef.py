import numpy as np 
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 
import time 
import warnings 

from .. import fem 
from ..ext import linalg 
from . import sn 
from . import qdf 
from .. import utils 

def WeakEddDivIntegrator(el1, el2, trans, qdf, qorder):
	elmat = np.zeros((el1.Nn*2, el2.Nn))
	ip, w = fem.quadrature.Get(qorder)

	for n in range(len(w)):
		vpgs = el1.CalcVPhysGradShape(trans, ip[n]) 
		s = el2.CalcShape(ip[n]) 
		E = qdf.EvalTensor(trans, ip[n]) 
		vpgsE = np.dot(vpgs.transpose(), E.flatten())
		linalg.AddOuter(-trans.Jacobian(ip[n])*w[n], vpgsE, s, elmat) 

	return elmat 

def MLBdrIntegrator(el1, el2, fi, qdf, qorder):
	elmat = np.zeros((el1.Nn*2, el2.Nn*2))
	ip, w = fem.quadrature.Get1D(qorder)

	for n in range(len(w)):
		xi1 = fi.ipt1.Transform(ip[n]) 
		G = qdf.EvalG(fi, ip[n]) 
		nor = fi.face.Normal(ip[n]) 
		E = qdf.EvalTensor(fi.trans1, xi1)
		vs = el1.CalcVShape(xi1)
		vEn = np.linalg.multi_dot([vs.transpose(), E, nor])
		nvs = np.dot(nor, vs)
		linalg.AddOuter(fi.face.Jacobian(ip[n])*w[n]/G, vEn, nvs, elmat)

	return elmat

def VEFInflowIntegrator(el, fi, qdf, qorder):
	elvec = np.zeros(2*el.Nn)
	ip, w = fem.quadrature.Get1D(qorder)

	for n in range(len(w)):
		xi1 = fi.ipt1.Transform(ip[n]) 
		nor = fi.face.Normal(ip[n])
		vs = el.CalcVShape(xi1)
		E = qdf.EvalTensor(fi.trans1, xi1)
		G = qdf.EvalG(fi, ip[n])
		Jin = qdf.EvalJinBdr(fi, ip[n])
		vEn = np.linalg.multi_dot([vs.transpose(), E, nor])
		elvec += 2*fi.face.Jacobian(ip[n])*w[n] * vEn / G * Jin 

	return elvec 

class AbstractVEF(sn.Sn):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None):
		sn.Sn.__init__(self, sweeper)
		self.phi_space = phi_space 
		self.J_space = J_space 
		self.lin_solver = lin_solver
		p = self.p = self.J_space.p

		self.sigma_a = lambda x: sweeper.sigma_t(x) - sweeper.sigma_s(x)

		self.Mt = fem.Assemble(self.J_space, fem.VectorMassIntegrator, sweeper.sigma_t, 2*p+1)
		self.Ma = fem.Assemble(self.phi_space, fem.MassIntegrator, self.sigma_a, 2*p+1)
		self.D = fem.MixAssemble(self.phi_space, self.J_space, fem.MixDivIntegrator, 1, 2*p+1)

		self.Q0 = np.zeros(self.phi_space.Nu)
		self.Q1 = np.zeros(self.J_space.Nu)
		for a in range(self.quad.N):
			Omega = self.quad.Omega[a] 
			w = self.quad.w[a] 
			self.Q0 += fem.AssembleRHS(phi_space, fem.DomainIntegrator, lambda x: sweeper.Q(x,Omega), 2*p+2) * w
			self.Q1 += fem.AssembleRHS(J_space, fem.VectorDomainIntegrator, lambda x: sweeper.Q(x,Omega)*Omega, 2*p+2) * w 

		self.qdf = qdf.QDFactors(self.space, self.quad, sweeper.psi_in) 

	def SourceIteration(self, psi, niter=50, tol=1e-10):
		phi = fem.GridFunction(self.phi_space)
		phi_old = fem.GridFunction(self.phi_space)
		for n in range(niter):
			start = time.time()
			phi_old.data = phi.data.copy()
			self.sweep.Sweep(psi, phi) 
			phi, J = self.Mult(psi)
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.LOUD):
				el = time.time() - start 
				print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(n+1, norm, el))

			if (norm < tol):
				break 

		if (norm > tol):
			warnings.warn('source iteration not converged. final tol = {:.3e}'.format(norm), utils.ToleranceWarning)

		return phi		

class VEF(AbstractVEF):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None):
		AbstractVEF.__init__(self, phi_space, J_space, sweeper, lin_solver)

	def Mult(self, psi):
		self.qdf.Compute(psi)
		G = fem.MixAssemble(self.J_space, self.phi_space, WeakEddDivIntegrator, self.qdf, 2*self.p+1)
		B = fem.BdrFaceAssemble(self.J_space, MLBdrIntegrator, self.qdf, 2*self.p+1)
		qin = fem.FaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf, 2*self.p+1)

		A = sp.bmat([[self.Mt+B, G], [self.D, self.Ma]])
		rhs = np.concatenate((self.Q1+qin, self.Q0))

		x = spla.spsolve(A.tocsc(), rhs) 
		phi = fem.GridFunction(self.phi_space)
		J = fem.GridFunction(self.J_space)
		phi.data = x[self.J_space.Nu:]
		J.data = x[:self.J_space.Nu]

		return phi, J 
