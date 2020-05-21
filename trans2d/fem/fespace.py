from . import mesh 
from . import element 
import numpy as np 
import sys 

class FESpace:
	def __init__(self, mesh, btype, p, vdim=1):
		self.mesh = mesh 
		self.p = p
		self.btype = btype 
		self.el = element.Element(btype, p)
		self.vdim = vdim 
		self.Ne = mesh.Ne 
		self.dofs = np.zeros((self.Ne, vdim*self.el.Nn), dtype=int)

	def Plot(self):
		import matplotlib.pyplot as plt 
		from matplotlib.patches import Polygon
		from matplotlib.collections import PatchCollection
		fig = plt.figure()
		for e in range(self.Ne):
			nodes = self.mesh.nodes[self.mesh.ele[e]]
			nodes[[2,3]] = nodes[[3,2]]
			poly = Polygon(nodes, fill=False)
			plt.gca().add_patch(poly)

		plt.plot(self.nodes[:,0], self.nodes[:,1], 'o')
		for n in range(self.nodes.shape[0]):
			plt.annotate(str(n), xy=(self.nodes[n,0], self.nodes[n,1]), 
				verticalalignment='bottom', horizontalalignment='left')

class H1Space(FESpace):
	def __init__(self, mesh, btype, p, vdim=1):
		FESpace.__init__(self, mesh, btype, p, vdim) 

		n_on_face = [np.arange(0,p+1), 
			np.arange(p, self.el.Nn+1, p+1), np.arange(p*(p+1), self.el.Nn), 
			np.arange(0, p*(p+1)+1, p+1)]
		if (p==1):
			sdofs = mesh.ele 
			c = np.max(sdofs)+1
		else:
			c = 0
			vis = np.zeros(self.Ne, dtype=bool) 
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

		self.nodes = np.zeros((c, 2))
		for e in range(self.Ne):
			trans = self.mesh.trans[e] 
			for n in range(self.el.Nn):
				self.nodes[self.dofs[e,n]] = trans.Transform(self.el.nodes[n])

	def LORefine(self):
		p = self.el.basis.p 
		ele = np.zeros((self.Ne*p**2, 4), dtype=int) 
		te = 0 
		for e in range(self.Ne):
			for i in range(p):
				for j in range(p):
					ele[te,0] = self.dofs[e,i*(p+1)+j]
					ele[te,1] = self.dofs[e,i*(p+1)+j+1] 
					ele[te,2] = self.dofs[e,(i+1)*(p+1)+j]
					ele[te,3] = self.dofs[e,(i+1)*(p+1)+j+1]
					te += 1 
		lomesh = mesh.AbstractMesh(self.nodes, ele, self.mesh.order)
		lospace = H1Space(lomesh, self.btype, 1, self.vdim) 
		return lospace 

class L2Space(FESpace):
	def __init__(self, mesh, btype, p, vdim=1, bfs=True):
		FESpace.__init__(self, mesh, btype, p, vdim)
		c = 0
		sdofs = np.zeros((self.Ne, self.el.Nn))
		if (p==0):
			for e in range(self.Ne):
				sdofs[e,0] = e 
			c = self.Ne 
		else:
			if (bfs):
				for v in self.mesh.graph.bfsiter(0):
					sdofs[v.index] = np.arange(c, self.el.Nn+c)
					c += self.el.Nn
			else:
				for e in range(self.Ne):
					sdofs[e] = np.arange(c, self.el.Nn+c)
					c += self.el.Nn 

		for e in range(self.Ne):
			vdofs = sdofs[e] 
			for d in range(1, self.vdim):
				vdofs = np.append(vdofs, sdofs[e]+c*d)

			self.dofs[e] = vdofs 

		self.Nu = self.vdim*c 

		self.nodes = np.zeros((c, 2))
		for e in range(self.Ne):
			trans = self.mesh.trans[e] 
			for n in range(self.el.Nn):
				self.nodes[self.dofs[e,n]] = trans.Transform(self.el.nodes[n])