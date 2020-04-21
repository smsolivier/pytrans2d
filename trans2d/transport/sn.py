import time 
import warnings 

from .. import fem 

class Sn:
	def __init__(self, sweep):
		self.sweep = sweep 
		self.space = self.sweep.space 
		self.LOUD = self.sweep.LOUD 
		self.p = self.sweep.space.p
		self.quad = sweep.quad

	def ComputeScalarFlux(self, psi):
		phi = fem.GridFunction(self.space)
		for a in range(self.quad.N):
			phi.data += self.quad.w[a]*psi.GetAngle(a).data

		return phi 

	def SourceIteration(self, psi, niter=50, tol=1e-10):
		phi = self.ComputeScalarFlux(psi)
		phi_old = fem.GridFunction(self.space)
		for n in range(niter):
			start = time.time()
			phi_old.data = phi.data.copy()
			self.sweep.Sweep(psi, phi) 
			phi = self.ComputeScalarFlux(psi)
			norm = phi.L2Diff(phi_old, 2*self.p+1)
			if (self.LOUD):
				el = time.time() - start 
				print('i={:3}, norm={:.3e}, {:.2f} s/iter'.format(n+1, norm, el))

			if (norm < tol):
				break 

		if (norm > tol):
			warnings.warn('source iteration not converged. final tol = {:.3e}'.format(norm), stacklevel=2)

		return phi