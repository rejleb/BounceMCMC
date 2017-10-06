import rebound
import ctypes
import numpy as np
import state

class RVVariations(object):
    def __init__(self, stat):
    	self.stat = stat
    	self.m_primary = 1.
    	self.G = 1.
        self.mcmc_vari1 = None
        self.mcmc_vari2 = None
        self.all_vari1 = None
        self.all_vari2 = None

    def calculate_all_variations(self):
    	res1 = np.zeros(len(self.stat.planets)*7)
    	a, lamb, k, h, ix, iy, m = 0.5, 0.0, 0.00, 0.00, 0.00, 0.00, 0.0005
    	p, q = reb_tools_solve_kepler_pal(h, k, lamb)
    	ahh = pyt_derivative_lambda(self.G, a, lamb, k, h, ix, iy, m, self.m_primary, p, q)
    	self.all_vari1 = ahh




path = '/home/reboundboks/Documents/rebound/src/librebound.so'
_lib = ctypes.CDLL(path)
_lib.reb_tools_solve_kepler_pal.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))

def reb_tools_solve_kepler_pal(h, k, l):
    global _lib
    p, q = ctypes.c_double(0.0), ctypes.c_double(0.0)
    _lib.reb_tools_solve_kepler_pal(ctypes.c_double(h), ctypes.c_double(k), ctypes.c_double(l), ctypes.byref(p), ctypes.byref(q))
    return p.value, q.value

def pyt_derivative_lambda(G, a, lamb, k, h, ix, iy, m, m_pri, p, q):
	dq_dlambda = -p/(1.-q)
	dp_dlambda = q/(1.-q)
	slp = np.sin(lamb+p)
	clp = np.cos(lamb+p)
	dclp_dlambda = -1./(1.-q)*slp
	dslp_dlambda = 1./(1.-q)*clp
	l = 1.-np.sqrt(1.-h*h-k*k)
	
	dxi_dlambda = a*(dclp_dlambda + dp_dlambda/(2.-l)*h)
	deta_dlambda = a*(dslp_dlambda - dp_dlambda/(2.-l)*k)
	iz = np.sqrt(np.abs(4.-ix*ix-iy*iy))
	dW_dlambda = deta_dlambda*ix-dxi_dlambda*iy

	x = dxi_dlambda+0.5*iy*dW_dlambda
	y = deta_dlambda-0.5*ix*dW_dlambda
	z = 0.5*iz*dW_dlambda

	an = np.sqrt(G*(m+m_pri)/a)
	ddxi_dlambda  = an/((1.-q)*(1.-q))*dq_dlambda*(-slp+q/(2.-l)*h) + an/(1.-q)*(-dslp_dlambda+dq_dlambda/(2.-l)*h)
	ddeta_dlambda = an/((1.-q)*(1.-q))*dq_dlambda*(+clp-q/(2.-l)*k) + an/(1.-q)*(dclp_dlambda-dq_dlambda/(2.-l)*k)
	ddW_dlambda = ddeta_dlambda*ix-ddxi_dlambda*iy

	vx = ddxi_dlambda+0.5*iy*ddW_dlambda
	vy = ddeta_dlambda-0.5*ix*ddW_dlambda
	vz = 0.5*iz*ddW_dlambda

	return x, y, z, vx, vy, vz