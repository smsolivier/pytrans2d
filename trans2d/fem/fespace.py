from . import mesh 
from . import element 
import numpy as np 
import sys 

class FESpace:
	def __init__(self, mesh, btype, p, vdim=1):
		self.mesh = mesh 
		self.p = p
		self.el = element.Element(btype, p)
		self.vdim = vdim 
		self.Ne = mesh.Ne 
		self.dofs = np.zeros((self.Ne, vdim*self.el.Nn), dtype=int)

class H1Space(FESpace):
	def __init__(self, mesh, btype, p, vdim=1):
		FESpace.__init__(self, mesh, btype, p, vdim) 

		c = 0
		vis = np.zeros(self.Ne, dtype=bool) 
		n_on_face = [np.arange(0,p+1), 
			np.arange(p, self.el.Nn+1, p+1), np.arange(p*(p+1), self.el.Nn), 
			np.arange(0, p*(p+1)+1, p+1)]
		sdofs = np.zeros((self.Ne, self.el.Nn))
		for v,d,p in self.mesh.graph.bfsiter(0, advanced=True):
			if (p==None):
				sdofs[v.index] = np.arange(0, self.el.Nn)
				c += self.el.Nn 

			else:
				new_dofs = np.zeros(self.el.Nn)-1 
				neigh = np.array([n.index for n in v.neighbors()])
				tovis = neigh[np.argwhere(vis[neigh]==True)[:,0]]
				for e in tovis:
					f1, f2 = self.mesh.GetOrientation(v.index, e)
					dofs = sdofs[e][n_on_face[f2]]
					new_dofs[n_on_face[f1]] = dofs 

				for i in range(len(new_dofs)):
					if (new_dofs[i]<0):
						new_dofs[i] = c 
						c += 1 
				sdofs[v.index] = new_dofs
			vis[v.index] = True 

		for e in range(self.Ne):
			vdofs = sdofs[e]
			for d in range(1, self.vdim):
				vdofs = np.append(vdofs, sdofs[e] + c*d) 

			self.dofs[e] = vdofs

		self.Nu = self.vdim*c

		self.bnodes = [] 
		for f in self.mesh.bface:
			self.bnodes += self.dofs[f.ElNo1, n_on_face[f.f1]].tolist()

		self.bnodes = np.unique(self.bnodes)

class L2Space(FESpace):
	def __init__(self, mesh, btype, p, vdim=1):
		FESpace.__init__(self, mesh, btype, p, vdim)
		c = 0
		sdofs = np.zeros((self.Ne, self.el.Nn))
		for v in self.mesh.graph.bfsiter(0):
			sdofs[v.index] = np.arange(c, self.el.Nn+c)
			c += self.el.Nn

		for e in range(self.Ne):
			vdofs = sdofs[e] 
			for d in range(1, self.vdim):
				vdofs = np.append(vdofs, sdofs[e]+c*d)

			self.dofs[e] = vdofs 

		self.Nu = self.vdim*c 