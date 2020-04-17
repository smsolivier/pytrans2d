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

	def ElementData(self):
		data = np.zeros(self.space.Ne)

		for e in range(self.space.Ne):
			data[e] = self.Interpolate(e, [0,0])

		return data 