import numpy as np 
from .. import fem 

class TVector:
	def __init__(self, space, quad):
		self.space = space 
		self.quad = quad 

		self.gf = [] 
		for a in range(self.quad.N):
			self.gf.append(fem.GridFunction(self.space))

	def Project(self, func):
		for a in range(self.quad.N):
			self.gf[a].Project(lambda x: func(x, self.quad.Omega[a]))

	def GetAngle(self, angle):
		return self.gf[angle]

	def SetAngle(self, angle, data):
		self.gf[angle].data = data

	def GetDof(self, angle, e):
		return self.gf[angle].GetDof(e)

	def SetDof(self, angle, e, vals):
		self.gf[angle].SetDof(e, vals) 
