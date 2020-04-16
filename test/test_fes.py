#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from trans2d import * 

mesh = RectMesh(3, 3)
space = H1Space(mesh, LagrangeBasis, 2, 2)

def test_fes():
	assert(space.Nu==98)
