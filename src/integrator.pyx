cdef double integrand(int n, double[3] args):
	x = args[0]
	A = args[1]
	b = args[2]

	return A*x*x + b
