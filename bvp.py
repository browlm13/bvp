#!/usr/bin/env python3

"""

    Construct linear system for 1D boundary value problem

	-u''(x) + c*u'(x) + d*u(x) = f(x), 0<x<1
	u(0) = u(1) = 0,
	
	domain [0,1]

	c,d - real number scalers corresponding to the strength 
		of convection and decay in the differential equation


    Use:
    	A, b = bvp(c, d, m, f)


	# super diagnol elements (-1/h**2 -c/2/h)
	# main diagnol elements : (2/h**2 + d)
	# sub diagnol elements : (-1/h**2 + c/2/h)
	# zeros everywhere else

"""

# external libs
import numpy as np

def bvp(c, d, m, f):

	# break into m-1 subintervals of size h
	h = 1.0/m
	xs = np.linspace(h,1.0-h,m-1) # x values to evaluate f at

	# construct vector b of size m-1 corresponding to f evaluated 
	# 1-dimensional numpy array
	vf = np.vectorize(f)
	b = vf(xs) # apply f to x values 

	# construct m-1 by m-1 symetric bandend matrix A
	A = np.zeros(shape=(m-1,m-1))

	# m = 1/h
	m2 = m**2 
	c_prime = m*c/2

	q = -m2 -c_prime 	# super diagnol elements
	r = 2*m2 +d 		# main diagnol elements
	s = -m2 + c_prime	# sub diagnol elements

	for j in range(0,m-2):
		i = j+1
		[i,j], A[j,j], A[j,i] = q, r, s

	A[-1,-1] = r

	# return A and b
	return A, b
