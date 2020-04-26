#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 

mesh = RectMesh(3, 3)
space = H1Space(mesh, LagrangeBasis, 2, 2)
l2 = L2Space(mesh, LagrangeBasis, 2, 2)

def test_fes():
	assert(space.Nu==98)

def test_l2():
	assert(l2.Nu==9*9*2)

def test_lorefine():
	p = 3
	space = H1Space(mesh, LagrangeBasis, p, 2)
	l2 = L2Space(mesh, LagrangeBasis, p-1)
	lospace = space.LORefine() 
	assert(lospace.Nu==space.Nu)
	assert(lospace.Ne==space.Ne*p**2) 
	lol2 = L2Space(lospace.mesh, LegendreBasis, 0)
	assert(lol2.Nu==l2.Nu)
