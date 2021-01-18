import numpy as np 
import scipy.sparse as sp 
import scipy.sparse.linalg as spla 
import time 
import warnings 

from . import sn 
from .. import fem 
from .. import utils 
from ..ext import linalg

def ModifiedInteriorPenaltyIntegrator(el1, el2, face, c, qorder):
	elmat = np.zeros((2*el1.Nn, 2*el2.Nn))
	ip, w = fem.quadrature.Get1D(qorder)
	p = el1.basis.p 

	j = np.zeros(el1.Nn+el2.Nn)
	a = np.zeros(el1.Nn+el2.Nn)
	for n in range(len(w)):
		xi1 = face.ipt1.Transform(ip[n])
		xi2 = face.ipt2.Transform(ip[n])
		c1 = c(face.trans1.Transform([0,0]))
		c2 = c(face.trans2.Transform([0,0]))
		nor = face.face.Normal(ip[n])

		s1 = el1.CalcShape(xi1)
		s2 = el2.CalcShape(xi2)
		pgs1 = el1.CalcPhysGradShape(face.trans1, xi1)
		pgs2 = el2.CalcPhysGradShape(face.trans2, xi2)
		j[:el1.Nn] = s1 
		j[el1.Nn:] = -s2 
		a[:el1.Nn] = c1*np.dot(nor, pgs1)
		a[el1.Nn:] = c2*np.dot(nor, pgs2)

		jac = face.face.Jacobian(ip[n])
		alpha = jac*w[n] 
		kappa_ip = p*(p+1)/2*(c1/face.trans1.Length() + c2/face.trans2.Length())*1000
		kappa = max(.25, kappa_ip)
		linalg.AddOuter(kappa*alpha, j, j, elmat)
		M = linalg.Outer((1 if face.boundary else .5)*alpha, j, a)
		sym = M + M.transpose()
		elmat -= sym 
	return elmat 

class MIP(sn.Sn):
	def __init__(self, sweep):
		sn.Sn.__init__(self, sweep)

		D = lambda x: 1/3/sweep.sigma_t(x) 
		sigma_a = lambda x: sweep.sigma_t(x) - sweep.sigma_s(x) 
		K = fem.Assemble(self.space, fem.DiffusionIntegrator, D, 2*self.p+1)
		F = fem.FaceAssembleAll(self.space, ModifiedInteriorPenaltyIntegrator, D, 2*self.p+1)
		Ma = fem.Assemble(self.space, fem.MassIntegrator, sigma_a, 2*self.p+1)
		self.lhs = (K + F + Ma).tocsc()
		self.lu = spla.splu(self.lhs)

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
			x = self.lu.solve(scat)
			phi.data += x
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.LOUD):
				el = time.time() - start 
				print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(n+1, norm, el))

			if (norm < tol):
				break 

		if (norm > tol):
			warnings.warn('source iteration not converged. final tol = {:.3e}'.format(norm), utils.ToleranceWarning)

		return phi		