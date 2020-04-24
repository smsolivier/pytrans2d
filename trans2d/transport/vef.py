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

def MLBdrIntegratorRowSum(el1, el2, fi, qdf, qorder):
	M = MLBdrIntegrator(el1, el2, fi, qdf, qorder)
	for d in range(2):
		for e in range(2):
			for i in range(el1.Nn):
				tmp = 0 
				for j in range(el1.Nn):
					tmp += M[d*el1.Nn + i, e*el1.Nn + j] 
					M[d*el1.Nn + i, e*el1.Nn + j] = 0 
				M[d*el1.Nn+i, e*el1.Nn+i] = tmp 

	return M 

def MLBdrIntegratorFullRowSum(el1, el2, fi, qdf, qorder):
	M = MLBdrIntegrator(el1, el2, fi, qdf, qorder)
	for i in range(M.shape[0]):
		tmp = 0
		for j in range(M.shape[1]):
			tmp += M[i,j]
			M[i,j] = 0 
		M[i,i] = tmp 
	return M 

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
		self.Mtl = fem.Assemble(self.J_space, fem.VectorMassIntegratorRowSum, sweeper.sigma_t, 2*p+1)
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
		self.linit = [] 

	def SourceIteration(self, psi, niter=50, tol=1e-10):
		phi = fem.GridFunction(self.phi_space)
		phi_old = fem.GridFunction(self.phi_space)
		for n in range(niter):
			start = time.time()
			phi_old.data = phi.data.copy()
			self.sweep.Sweep(psi, phi) 
			phi, J = self.Mult(psi)
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.lin_solver!=None):
				self.linit.append(self.lin_solver.it)
			if (self.LOUD):
				el = time.time() - start 
				if (self.lin_solver!=None):
					print('i={:3}, norm={:.3e}, {:.2f} s/iter, {} linear iters'.format(
						n+1, norm, el, self.lin_solver.it))
				else:
					print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(
						n+1, norm, el))

			if (norm < tol):
				break 

		if (self.LOUD and self.lin_solver!=None):
			self.avg_linit = np.mean(self.linit) 
			print('avg linear iters = {:.2f}'.format(self.avg_linit))

		if (norm > tol):
			warnings.warn('source iteration not converged. final tol = {:.3e}'.format(norm), utils.ToleranceWarning)

		return phi		

class VEF(AbstractVEF):
	def __init__(self, phi_space, J_space, sweeper, lin_solver=None):
		AbstractVEF.__init__(self, phi_space, J_space, sweeper, lin_solver)
		self.full_lump = False
		self.direct_inv = False 

	def Mult(self, psi):
		self.qdf.Compute(psi)
		G = fem.MixAssemble(self.J_space, self.phi_space, WeakEddDivIntegrator, self.qdf, 2*self.p+1)
		B = fem.BdrFaceAssemble(self.J_space, MLBdrIntegrator, self.qdf, 2*self.p+1)
		qin = fem.FaceAssembleRHS(self.J_space, VEFInflowIntegrator, self.qdf, 2*self.p+1)

		A = self.Mt + B
		rhs = np.concatenate((self.Q1+qin, self.Q0))

		if (self.lin_solver==None):
			M = sp.bmat([[A, G], [self.D, self.Ma]])
			x = spla.spsolve(M.tocsc(), rhs) 
		else:
			if not(self.direct_inv):
				Al = fem.AssembleBlocks(self.J_space, fem.VectorMassIntegratorRowSum, self.sweep.sigma_t, 2*self.p+1)
				Bl = fem.BdrFaceAssembleBlocks(self.J_space, 
					MLBdrIntegratorFullRowSum if self.full_lump else MLBdrIntegratorRowSum, self.qdf, 2*self.p+1)
				Al += Bl 
				a = Al[0,0].diagonal()
				b = Al[0,1].diagonal()
				c = Al[1,0].diagonal()
				d = Al[1,1].diagonal()
				w = 1/(a - b/d*c)
				x = -1/a*b*w
				z = 1/(d - c/a*b)
				y = -1/d*c*z 
				Alinv = sp.bmat([[sp.diags(w), sp.diags(x)], [sp.diags(y), sp.diags(z)]])
			else:
				Alinv = spla.inv(A) 

			x = self.lin_solver.Solve(A, Alinv, G, self.D, self.Ma, rhs) 

		phi = fem.GridFunction(self.phi_space)
		J = fem.GridFunction(self.J_space)
		phi.data = x[self.J_space.Nu:]
		J.data = x[:self.J_space.Nu]

		return phi, J 
