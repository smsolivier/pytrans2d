import numpy as np 
import scipy.sparse.linalg as spla 

from .. import fem 

class AbstractSweeper:
	def __init__(self, space, quad, sigma_t, sigma_s, Q, psi_in, LOUD=True):
		self.space = space 
		self.quad = quad 
		self.sigma_t = sigma_t 
		self.sigma_s = sigma_s 
		self.Q = Q 
		self.psi_in = psi_in 
		self.LOUD = LOUD 

	def FormScattering(self, phi):
		scat = fem.GridFunction(self.space)
		if (self.Ms.shape[1]!=phi.space.Nu):
			self.Ms = fem.MixAssemble(self.space, phi.space, fem.MixMassIntegrator, self.sigma_s, 2*self.space.p+1)
		scat.data = self.Ms * phi.data / (4*np.pi) 
		return scat 

class DirectSweeper(AbstractSweeper):
	def __init__(self, space, quad, sigma_t, sigma_s, Q, psi_in, LOUD=True):
		AbstractSweeper.__init__(self, space, quad, sigma_t, sigma_s, Q, psi_in, LOUD) 

		p = self.space.p
		self.Mt = fem.Assemble(self.space, fem.MassIntegrator, self.sigma_t, 2*p+1)
		self.Ms = fem.Assemble(self.space, fem.MassIntegrator, self.sigma_s, 2*p+1)
		Gx = fem.Assemble(self.space, fem.WeakConvectionIntegrator, np.array([1,0]), 2*p+1)
		Gy = fem.Assemble(self.space, fem.WeakConvectionIntegrator, np.array([0,1]), 2*p+1)
		self.LHS = []
		self.RHS = []
		self.lu = []
		for n in range(self.quad.N):
			Omega = self.quad.Omega[n]
			G = Omega[0]*Gx + Omega[1]*Gy
			F = fem.FaceAssembleAll(self.space, fem.UpwindTraceIntegrator, Omega, 2*p+1)
			I = fem.FaceAssembleRHS(self.space, fem.InflowIntegrator, [Omega, self.psi_in], 2*p+1)
			b = fem.AssembleRHS(self.space, fem.DomainIntegrator, lambda x: self.Q(x,Omega), 2*p+1)

			self.LHS.append(G + F + self.Mt)
			self.RHS.append(b + I)
			self.lu.append(spla.splu(self.LHS[-1]))

	def Sweep(self, psi, phi):
		scat = self.FormScattering(phi)
		for a in range(self.quad.N):
			angle = self.lu[a].solve(self.RHS[a] + scat.data)
			psi.SetAngle(a, angle)


