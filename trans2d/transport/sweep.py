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

		self.Ms = fem.Assemble(self.space, fem.MassIntegrator, self.sigma_s, 2*self.space.p+1)

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

	def ComputeResidual(self, psi, phi):
		res = 0 
		scat = self.FormScattering(phi)
		for a in range(self.quad.N):
			res_angle = self.LHS[a]*psi.GetAngle(a).data - scat.data - self.RHS[a]
			res += np.dot(res_angle, res_angle)

		return np.sqrt(res)

class Sweeper(AbstractSweeper):
	def __init__(self, space, quad, sigma_t, sigma_s, Q, psi_in, LOUD=True):
		AbstractSweeper.__init__(self, space, quad, sigma_t, sigma_s, Q, psi_in, LOUD) 

		mesh = self.space.mesh 
		self.graphs = []
		for a in range(self.quad.N):
			self.graphs.append(mesh.graph.copy()) 
		nbel = len(mesh.bel)
		self.start = np.zeros(self.quad.N, dtype=int)
		for a in range(self.quad.N):
			ninflow = np.zeros(self.space.Ne, dtype=int)
			Omega = self.quad.Omega[a]
			for face in mesh.bface:
				nor = face.face.Normal(0)
				dot = np.dot(nor, Omega)
				if (dot<0):
					ninflow[face.ElNo1] += 1 

			self.graphs[a].vs['ninflow'] = ninflow.tolist()
			self.start[a] = self.graphs[a].vs(ninflow_eq=2)[0].index

		p = self.p = self.space.p 
		self.Mt = [] 
		self.Gx = [] 
		self.Gy = [] 
		for e in range(self.space.Ne):
			trans = self.space.mesh.trans[e]
			self.Mt.append(fem.MassIntegrator(self.space.el, trans, self.sigma_t, 2*p+1))
			self.Gx.append(fem.WeakConvectionIntegrator(self.space.el, trans, np.array([1,0]), 2*p+1))
			self.Gy.append(fem.WeakConvectionIntegrator(self.space.el, trans, np.array([0,1]), 2*p+1))

		self.LHS = np.zeros((self.quad.N, self.space.Ne, self.space.el.Nn, self.space.el.Nn))
		self.LHSI = np.zeros((self.quad.N, self.space.Ne, self.space.el.Nn, self.space.el.Nn))
		self.RHS = np.zeros((self.quad.N, self.space.Ne, self.space.el.Nn))
		self.I = np.zeros((self.quad.N, self.space.Ne, 4,self.space.el.Nn, self.space.el.Nn))
		el = self.space.el
		for a in range(self.quad.N):
			Omega = self.quad.Omega[a]
			for v in self.graphs[a].bfsiter(int(self.start[a])):
				lhs = self.Mt[v.index] + Omega[0]*self.Gx[v.index] + Omega[1]*self.Gy[v.index] 
				rhs = fem.DomainIntegrator(self.space.el, self.space.mesh.trans[v.index], 
					lambda x: self.Q(x,Omega), 2*p+1)
				for f in range(4):
					iidx = mesh.iface2el[v.index,f] 
					bidx = mesh.bface2el[v.index,f] 

					if (iidx>=0):
						fi = mesh.iface[iidx]
						orient = True if fi.ElNo1==v.index else False 
						sgn = 1 if orient else -1
						nor = fi.face.Normal(0) * sgn 
						dot = np.dot(Omega, nor) 
						F = fem.UpwindTraceIntegrator(el, el, fi, Omega, 2*self.p+1)
						if (dot>0):
							subF = F[:el.Nn,:el.Nn] if orient else F[el.Nn:,el.Nn:]
							lhs += subF
						else:
							subF = F[:el.Nn,el.Nn:] if orient else F[el.Nn:,:el.Nn]
							self.I[a,v.index,f] = subF 
					else:
						fi = mesh.bface[bidx]
						nor = fi.face.Normal(0)
						dot = np.dot(Omega, nor)
						if (dot>0):
							F = fem.UpwindTraceIntegrator(el, el, fi, Omega, 2*self.p+1)
							lhs += F[:el.Nn,:el.Nn]
						else:
							I = fem.InflowIntegrator(el, fi, [Omega, self.psi_in], 2*self.p+1)
							rhs += I 		

				self.LHS[a,v.index] = lhs 
				self.LHSI[a,v.index] = np.linalg.inv(lhs)
				self.RHS[a,v.index] = rhs 

	def Sweep(self, psi, phi):
		scat = self.FormScattering(phi)
		mesh = self.space.mesh
		el = self.space.el 
		p = self.p
		for a in range(self.quad.N):
			Omega = self.quad.Omega[a]
			angle = psi.GetAngle(a)
			for v in self.graphs[a].bfsiter(int(self.start[a])):
				lhs = self.LHS[a,v.index]
				rhs = self.RHS[a,v.index] + scat.GetDof(v.index)
				for f in range(4):
					iidx = mesh.iface2el[v.index,f] 
					bidx = mesh.bface2el[v.index,f] 

					if (iidx>=0):
						fi = mesh.iface[iidx]
						orient = True if fi.ElNo1==v.index else False 
						sgn = 1 if orient else -1
						nor = fi.face.Normal(0) * sgn 
						dot = np.dot(Omega, nor) 
						if (dot<0):
							subF = self.I[a,v.index,f]
							upw = fi.ElNo2 if orient else fi.ElNo1
							rhs -= np.dot(subF, angle.GetDof(upw)) 
				x = np.dot(self.LHSI[a,v.index], rhs)
				angle.SetDof(v.index, x) 

	def ComputeResidual(self, psi, phi):
		scat = self.FormScattering(phi)
		mesh = self.space.mesh
		el = self.space.el 
		p = self.p
		res = 0
		for a in range(self.quad.N):
			Omega = self.quad.Omega[a]
			angle = psi.GetAngle(a)
			for v in self.graphs[a].bfsiter(int(self.start[a])):
				lhs = self.LHS[a,v.index]
				rhs = self.RHS[a,v.index] + scat.GetDof(v.index)
				for f in range(4):
					iidx = mesh.iface2el[v.index,f] 
					bidx = mesh.bface2el[v.index,f] 

					if (iidx>=0):
						fi = mesh.iface[iidx]
						orient = True if fi.ElNo1==v.index else False 
						sgn = 1 if orient else -1
						nor = fi.face.Normal(0) * sgn 
						dot = np.dot(Omega, nor) 
						if (dot<0):
							subF = self.I[a,v.index,f]
							upw = fi.ElNo2 if orient else fi.ElNo1
							rhs -= np.dot(subF, angle.GetDof(upw)) 
				diff = np.dot(lhs, angle.GetDof(v.index)) - rhs 
				res += np.dot(diff, diff)

		return np.sqrt(res)