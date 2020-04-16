#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 
from pytest import approx 

mesh = RectMesh(3, 3)

def test_trans():
	trans = mesh.trans[0]
	assert(trans.Transform([-1,-1])==approx([0,0]))
	assert(trans.Transform([1,1])==approx([1/3,1/3]))
	assert(trans.Transform([0,-1])==approx([1/6,0]))

def test_jacobian():
	trans = mesh.trans[0]
	assert(trans.Jacobian([0,0])==approx(1/36))

def test_boundary():
	assert(mesh.bel==[0,1,2,3,5,6,7,8])

def test_inverse():
	trans = mesh.trans[0]
	assert(trans.InverseMap([1/6, 1/6])==approx([0,0]))
	assert(trans.InverseMap([1/3,1/3])==approx([1,1]))
	assert(trans.InverseMap([1/3,1/6])==approx([1,0]))