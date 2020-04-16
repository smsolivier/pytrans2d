#include "Python.h"
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> 

static PyObject* _outer(PyObject* self, PyObject* args) {
	PyArrayObject *v1, *v2; 
	if (!PyArg_ParseTuple(args, "OO", &v1, &v2)) return NULL; 

	int m = PyArray_DIM(v1, 0); 
	int n = PyArray_DIM(v2, 0); 

	double *pv1 = PyArray_DATA(v1); 
	double *pv2 = PyArray_DATA(v2); 
	double* mat = malloc(sizeof(double)*m*n); 
	const npy_intp dims[2] = {m, n}; 
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			mat[j+i*n] = pv1[i] * pv2[j]; 
		}
	}
	return PyArray_SimpleNewFromData(2, &dims[0], NPY_DOUBLE, mat); 
}

static PyObject* _AddMultVtW(PyObject* self, PyObject* args) {
	PyArrayObject *v, *w, *elmat; 
	double alpha; 
	if (!PyArg_ParseTuple(args, "dOOO", &alpha, &v, &w, &elmat)) return NULL; 

	int m = PyArray_DIM(v, 0); 
	int n = PyArray_DIM(w, 0); 

	double *pv = PyArray_DATA(v); 
	double *pw = PyArray_DATA(w); 
	double *pm = PyArray_DATA(elmat); 
	const npy_intp dims[] = {m, n}; 
	for (int i=0; i<m; i++) {
		for (int j=0; j<n; j++) {
			pm[j+i*n] += pv[i] * pw[j] * alpha; 
		}
	}
	return Py_None; 
}

static PyMethodDef mainMethods[] = {
	{"Outer", _outer, METH_VARARGS, "outer product of two vectors"}, 
	{"AddMultVtW", _AddMultVtW, METH_VARARGS, "outer product of two row vectors multiplied by a constant"}, 
	{NULL, NULL, 0, NULL}
}; 

static PyModuleDef mod = {PyModuleDef_HEAD_INIT, "linalg", 
	"module for c extensions that numpy should already have", -1, mainMethods}; 

PyMODINIT_FUNC PyInit_linalg(void) {
	import_array(); 
	return PyModule_Create(&mod); 
}