import numpy as np 
from .quadrature import quadrature

class GridFunction:
	def __init__(self, space):
		self.space = space 
		self.data = np.zeros(self.space.Nu)

	def GetDof(self, e):
		return self.data[self.space.dofs[e]]

	def SetDof(self, e, vals):
		self.data[self.space.dofs[e]] = vals 

	def Interpolate(self, e, xi):
		return self.space.el.Interpolate(xi, self.GetDof(e))

	def InterpolateGradient(self, e, xi):
		trans = self.space.mesh.trans[e]
		return self.space.el.InterpolateGradient(trans, xi, self.GetDof(e))

	def Project(self, func):
		for e in range(self.space.Ne):
			trans = self.space.mesh.trans[e]
			vals = np.zeros(self.space.el.Nn)
			for i in range(self.space.el.Nn):
				X = trans.Transform(self.space.el.nodes[i])
				vals[i] = func(X)

			self.SetDof(e, vals)

	def ProjectGF(self, gf):
		for e in range(self.space.Ne):
			vals = np.zeros(self.space.el.Nn)
			for i in range(self.space.el.Nn):
				vals[i] = gf.Interpolate(e, self.space.el.nodes[i])

			self.SetDof(e, vals) 			

	def L2Error(self, ex, qorder):
		ip, w = quadrature.Get(qorder)
		l2 = 0 
		for e in range(self.space.Ne):
			el = self.space.el 
			trans = self.space.mesh.trans[e]
			for n in range(len(w)):
				X = trans.Transform(ip[n])
				exact = np.array(ex(X))
				fem = self.Interpolate(e, ip[n])
				diff = exact - fem 

				l2 += np.dot(diff, diff) * w[n] * trans.Jacobian(ip[n])

		return np.sqrt(l2) 

	def L2ProjError(self, ex, qorder):
		gf = GridFunction(self.space)
		gf.Project(ex)
		return self.L2Diff(gf, qorder)

	def L2Diff(self, gf, qorder):
		ip, w = quadrature.Get(qorder)
		l2 = 0 
		for e in range(self.space.Ne):
			trans = self.space.mesh.trans[e]
			for n in range(len(w)):
				this = self.Interpolate(e, ip[n])
				that = gf.Interpolate(e, ip[n])
				l2 += np.dot(this-that, this-that) * w[n] * trans.Jacobian(ip[n])
		return np.sqrt(l2)

	def L2Norm(self, qorder):
		ip, w = quadrature.Get(qorder)
		l2 = 0 
		for e in range(self.space.Ne):
			trans = self.space.mesh.trans[e] 
			for n in range(len(w)):
				val = self.Interpolate(e, ip[n]) 
				l2 += np.dot(val, val) * w[n] * trans.Jacobian(ip[n]) 

		return np.sqrt(l2) 

	def ElementData(self):
		if (self.space.vdim==1):
			data = np.zeros(self.space.Ne)

			for e in range(self.space.Ne):
				data[e] = self.Interpolate(e, [0,0])
		else:
			data = np.zeros((self.space.Ne, 2))
			for e in range(self.space.Ne):
				data[e] = self.Interpolate(e, [0,0])

		return data 

	def NodeData(self):
		ref_geom = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])
		if (self.space.vdim==1):
			data = np.zeros((self.space.Ne, 4))
			for e in range(self.space.Ne):
				for n in range(4):
					data[e,n] = self.Interpolate(e, ref_geom[n])

		else:
			data = np.zeros((self.space.Ne, 4, 2))
			for e in range(self.space.Ne):
				for n in range(4):
					data[e,n] = self.Interpolate(e, ref_geom[n])

		return data 			