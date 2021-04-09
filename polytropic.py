import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import optimize
#from tqdm import tqdm
from tqdm.notebook import tqdm
import warnings


#matplotlib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['xtick.direction'] = 'in'
matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.top'] = True
matplotlib.rcParams['ytick.right'] = True
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams["figure.dpi"] = 100
#matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

"""
Solve for polytropic with rotation
"""

def model_odes(x, y, n, A=0., fA=None, qA=None):
    if fA is not None:
        return y[1], -2/x*y[1] - y[0]**n + 3*fA(x) + qA(x)
    else:
        return y[1], -2/x*y[1] - y[0]**n + 3*A

def model_initials(x, n, A=0., fA=None, qA=None):
    if fA is not None:
        A = fA(0.)
    if A==0.:
        y = 1 - x**2/6 + n*x**4/120 - n*(8*n-5)*x**6/15120
        dy = - x/3 + n*x**3/30 - n*(8*n-5)*x**5/2520
    else:
        y = 1 - (1-3*A)*x**2/6 + (1-3*A)*n*x**4/120
        dy = - (1-3*A)*x/3 + (1-3*A)*n*x**3/30
    return y+0j, dy+0j

def model_solutions(n, A=0., x0=-1e-12, x1=50, num=1000000, y0=(np.nan, np.nan), method='RK45', rtol=1e-6, atol=1e-6, fA=None, qA=None):
    t = np.linspace(x0, x1, num=num)
    if np.nan in y0:
        y0 = model_initials(x0, n, A, fA, qA)
    sol = integrate.solve_ivp(model_odes, (x0, x1), y0, t_eval = t, args=(n, A, fA, qA), method=method, rtol=rtol, atol=atol)
    i = 0
    for y in np.real(sol.y[0]):
        if y<0:
            #print(y)
            break
        i += 1
    #print(i, y)
    imax = i + 10
    if imax >= num:
        x1 = np.nan
        imax = num-1
    else:
        x1 = sol.t[i-1]/2+ sol.t[i]/2
        x1 = sol.t[i-1]
    return sol, imax, x1

def find_x1(theta, theta_p, x1, n_iter):
    for i in range(n_iter):
        #print(i, x1)
        x1 = x1 - theta(x1)/theta_p(x1)
    return x1

def find_Om2(theta, theta_p, x1, A, fA=None):
    if fA is not None:
        A = fA(x1)
    M = -x1**2*theta_p(x1) + A*x1**3
    Om2 = x1**3*A/M
    return Om2

def find_critical_A(n, A_l=0., A_h=1., eps=1e-12):
    while A_h - A_l > eps:
        A_m = (A_h+A_l)/2
        sol, imax, x1_m = model_solutions(n, A=A_m)
        if np.isnan(x1_m):
            A_h = A_m
        else:
            A_l = A_m
        #print('%.12e'%A_m)
    return A_l

def theta_interp(sol, imax, x1):
    imax += 10
    if imax >= len(sol.t):
        imax = len(sol.t)
    theta = interpolate.interp1d(sol.t[:imax], np.real(sol.y[0])[:imax], fill_value='extrapolate')
    theta_p = interpolate.interp1d(sol.t[:imax], np.real(sol.y[1])[:imax], fill_value='extrapolate')
    ypp = np.diff(np.real(sol.y[1])[:imax])/np.diff(sol.t[:imax])
    theta_pp = interpolate.interp1d(sol.t[:imax-1], ypp, fill_value='extrapolate')
    x1 = find_x1(theta, theta_p, x1, 12)
    return theta, theta_p, theta_pp, x1


"""
Find first order approximation
"""
def t1_odes(x, y, n, theta):
    return y[1], -2/x*y[1] - y[0]*n*theta(x)**(n-1) + 3

def find_theta1(n, theta, x1, x0=-1e-12, num=100000):
    t = np.linspace(x0, x1, num=num)
    sol = integrate.solve_ivp(t1_odes, (x0, x1), (0.,0.), t_eval = t, args=(n, theta))
    theta1 = interpolate.interp1d(sol.t, np.real(sol.y[0]), fill_value='extrapolate')
    theta1_p = interpolate.interp1d(sol.t, np.real(sol.y[1]), fill_value='extrapolate')
    return theta1, theta1_p







class poly:
    def __init__(self, n, A=0., xmax=15., fA=None, qA=None):
        sol, imax, x1 = model_solutions(n, x0=-1e-12, x1=xmax, rtol=1e-8, atol=1e-8, A=A, fA=fA, qA=qA)
        #print(sol.t, sol.y, x1)
        self.theta, self.theta_p, self.theta_pp, self.x1 = theta_interp(sol, imax, x1)
        self.Om2 = find_Om2(self.theta, self.theta_p, self.x1, A, fA)
        if fA is not None:
            self.fOm2 = lambda x: fA(x)/fA(x1)*self.Om2
            self.A = fA(x1)
            self.fA = fA
            self.qA = qA
        else:
            fOm2 = lambda x: self.Om2
            self.fOm2 = np.vectorize(fOm2)
            self.A = A
            self.fA = np.vectorize(lambda x: A)
            self.qA = np.vectorize(lambda x: 0.)
        self.n = n
        
        self.theta1, self.theta1_p = find_theta1(n, self.theta, self.x1)

        x = np.linspace(0, x1, num=1000)
        mr2Om_int = np.sum(x**2 *(self.theta(x))**n *x**2 *self.fOm2(x)*2/3)
        MR2 = np.sum(x**2 *(self.theta(x))**n *x**2 *2/3)
        self.J = mr2Om_int/MR2

        self.q = lambda x: self.qA(x) /2./self.fA(x)


    def V(self, x):
        x = max(x, 1e-10)
        return (1+self.n)*(-self.theta_p(x)+self.fA(x)*x)/self.theta(x) *x

    def c_1(self, x):
        x = max(x, 1e-10)
        return (-self.theta_p(self.x1)/self.x1 + self.fA(self.x1)) /(-self.theta_p(x)/x + self.fA(x))




const_G = 6.67428e-8
const_Na = 6.02e23
const_mu = 1
const_k = 1.380649e-16


class polystar:
    def __init__(self, p, M, R, L=1., gamma=5./3., f_Om=0):
        theta = p.theta
        theta_p = p.theta_p
        x1 = p.x1
        A = p.A
        fA = p.fA
        qA = p.qA
        fOm2 = p.fOm2
        if fA is not None:
            A = fA(x1)
        else:
            fA = lambda r: A*r**0
        n = p.n
        mass_factor = -theta_p(x1)/x1+A

        self.p = p
        self.M = M
        self.R = R
        self.L = L
        self.gamma = gamma
        self.f_Om = f_Om

        self.rho_c = M/(4*np.pi*R**3*mass_factor)
        self.p_c = 1./(4*np.pi*(n+1)*(mass_factor*x1)**2)*const_G*M**2/R**4
        self.K = self.p_c / self.rho_c**(1+1/n)
        self.r_n = np.sqrt((n+1)*self.p_c/(4*np.pi*const_G*self.rho_c**2))
        self.T_c = 1/((n+1)*(mass_factor*x1**2)) *const_G*const_mu/(const_Na*const_k) *M/R

        self.xi = lambda r: r/self.r_n
        self.rho_r = lambda r: self.rho_c*theta(self.xi(r))**n
        self.drho_dr = lambda r: self.rho_c* n*theta(self.xi(r))**(n-1) * theta_p(self.xi(r)) /self.r_n
        self.M_r = lambda r: 4*np.pi*self.r_n**3*self.rho_c*(-self.xi(r)**2*theta_p(self.xi(r)) + fA(self.xi(r))*self.xi(r)**3)
        self.g_r = lambda r: const_G*self.M_r(r)/r**2
        self.p_r = lambda r: self.K*self.rho_r(r)**(1+1/n)
        self.dp_dr = lambda r: self.K* (1+1/n)*self.rho_r(r)**(1/n) *self.drho_dr(r)
        self.T_r = lambda r: self.T_c*theta(self.xi(r))
        self.N2_r = lambda r, Gamma_1: self.g_r(r)*(1/Gamma_1 *self.dp_dr(r)/self.p_r(r)-self.drho_dr(r)/self.rho_r(r) )
        self.c1_r = lambda r: r**3*M/(R**3*self.M_r(r))
        self.Omega2_r = lambda r: fOm2(self.xi(r))*const_G*M/R**3
        self.dlnOmega_dlnr_r = lambda r: np.nan_to_num(qA(self.xi(r)) /2./fA(self.xi(r)))

    def write_mesa(self, fname, n_grids=2000, ver=233, eps_l=1e-3, eps_r = 1e-3):
        gamma = self.gamma
        n = self.p.n 
        R = self.R
        M = self.M
        Omega2 = self.p.Om2
        f_Om = self.f_Om



        # use equally spaced x = r/R
        k_mesa = np.arange(n_grids)+1
        r_mesa = np.linspace(eps_l*R, (1-eps_r)*R, num=n_grids)
        M_r_mesa = self.M_r(r_mesa)
        L_r_mesa = np.repeat(1., n_grids)
        p_mesa = self.p_r(r_mesa)
        T_mesa = self.T_r(r_mesa)
        rho_mesa = self.rho_r(r_mesa)
        nabla_mesa = np.repeat(1/(1+n), n_grids)
        N2_mesa = self.N2_r(r_mesa, gamma)
        Gamma_1_mesa = np.repeat(gamma, n_grids)
        nabla_ad_mesa = nabla_mesa
        delta_mesa = np.repeat(n, n_grids)
        kappa_mesa = np.repeat(1., n_grids)
        kappa_kappa_T_mesa = np.repeat(1., n_grids)
        kappa_kappa_rho_mesa = np.repeat(1., n_grids)
        epsilon_nuc_grav_mesa = np.repeat(1., n_grids)
        epsilon_nuc_T_mesa = np.repeat(1., n_grids)
        epsilon_nuc_rho_mesa = np.repeat(1., n_grids)
        Omega_mesa = np.zeros((n_grids))
        #Omega_mesa = np.repeat(np.sqrt(Omega2*const_G*M/R**3), n_grids)
        Omega_mesa = np.sqrt(self.Omega2_r(r_mesa))

        #print(Omega_mesa)

        if ver == 233:
            f_Omega = np.repeat(f_Om, n_grids)
            dlnOmega_dlnr = self.dlnOmega_dlnr_r(r_mesa)
            nu_viscosity = np.repeat(1e30, n_grids)
            dlnnu_dlnr = np.repeat(0., n_grids)

        if ver == 100:
            data_mesa = np.transpose([
                k_mesa, 
                r_mesa,
                M_r_mesa,
                L_r_mesa,
                p_mesa,
                T_mesa,
                rho_mesa,
                nabla_mesa,
                N2_mesa,
                Gamma_1_mesa,
                nabla_ad_mesa,
                delta_mesa,
                kappa_mesa,
                kappa_kappa_T_mesa,
                kappa_kappa_rho_mesa,
                epsilon_nuc_grav_mesa,
                epsilon_nuc_T_mesa,
                epsilon_nuc_rho_mesa,
                Omega_mesa,])
        elif ver == 233:
            data_mesa = np.transpose([
                k_mesa, 
                r_mesa,
                M_r_mesa,
                L_r_mesa,
                p_mesa,
                T_mesa,
                rho_mesa,
                nabla_mesa,
                N2_mesa,
                Gamma_1_mesa,
                nabla_ad_mesa,
                delta_mesa,
                kappa_mesa,
                kappa_kappa_T_mesa,
                kappa_kappa_rho_mesa,
                epsilon_nuc_grav_mesa,
                epsilon_nuc_T_mesa,
                epsilon_nuc_rho_mesa,
                Omega_mesa,
                dlnOmega_dlnr,
                f_Omega,
                nu_viscosity,
                dlnnu_dlnr,
            ])
        data_mesa = np.nan_to_num(data_mesa)

        #print(Omega_mesa)

        #write file
        fmt = "     ".join(["%6d"] + ["%.16e"] * (data_mesa.shape[1]-1))

        with open(fname, 'w') as f:
            f.write('%6d     %.16e     %.16e     %.16e     %d\n'%(n_grids, self.M, self.R, self.L, ver))
            np.savetxt(f, data_mesa, fmt=fmt)


"""
Solving for oscialltion eigen functions and eigenvalues
"""
def osc_rot_odes(x, y, w2, Om2, n, theta, theta_p, x1, f_Om, gamma, A, fA=None, qA=None, fOm2=None):
    if fA is not None:
        A = fA(x)
        Om2 = fOm2(x)
        A1 = fA(x1)
    else:
        A1 = A
    res1 = -1/x*(3*y[0] + 1/gamma*y[1])
    r3M = -theta_p(x1)/x1 + A1
    r3M /= -theta_p(x)/x + A
    # print(x, A, A1)
    fac0 = 4 + ((2*f_Om-1)*Om2+w2) *r3M
    fac1 = 1 - Om2 *r3M
    res2 = (1+n)*(-theta_p(x)+A*x)/theta(x)*(fac0*y[0]+fac1*y[1])
    return res1, res2

def osc_rot_bc_x1(y0, y1, w2, Om2, f_Om):
    return (4+(2*f_Om-1)*Om2+w2)*y0+ (1-Om2)* y1

def osc_rot_check_bc(w2, Om2, n, theta, theta_p, x0=1e-5, x1=4.3, num=10000, f_Om=0., gamma=5./3., A=0., method='RK45', fA=None, qA=None, fOm2=None):
    t = np.linspace(x0, x1, num=num)
    y0 = (1./3., -gamma)
    sol = integrate.solve_ivp(osc_rot_odes, (x0, x1), y0, t_eval = t, args=(w2, Om2, n, theta, theta_p, x1, f_Om, gamma, A, fA, qA, fOm2), method=method)
    dzeta = sol.y[0][-1]
    dp = sol.y[1][-1]
    return osc_rot_bc_x1(dzeta, dp, w2, Om2, f_Om)

def osc_rot_solutions(n, theta, theta_p, x0=1e-5, x1=4.3, num=100000, w2_l=2, w2_h=6, Om2=0, f_Om=0, gamma=5./3., A=0., method='RK45', tol_b=1e-3, tol_w2=1e-12, verbose=True, fA=None, qA=None, fOm2=None):
    t = np.linspace(x0, x1, num=num)
    y0 = (1./3., -gamma)
    p = 0.5
    eps = tol_b+1.0
    i = 0
    if fOm2 is not None:
        Om2 = fOm2(x1)
    while eps>tol_b:
        w2_m = w2_l*p + w2_h*(1-p)
        w2s = [w2_l, w2_m, w2_h]
        if verbose:
            print('Loop %s, eps=%e,'%(i, eps), w2s)
        if w2_h-w2_l < tol_w2:
            print('Break as tol_w2 reached')
            break
        
        bcs = []
        #print('----', w2s)
        for w2 in w2s:
            sol = integrate.solve_ivp(osc_rot_odes, (x0, x1), y0, t_eval = t, args=(w2, Om2, n, theta, theta_p, x1, f_Om, gamma, A, fA, qA, fOm2), method=method, atol=1e-5)
            dzeta = sol.y[0][-1]
            dp = sol.y[1][-1]
            bcs.append(osc_rot_bc_x1(dzeta, dp, w2, Om2, f_Om))
        eps = np.abs(bcs[1])
        if bcs[0]*bcs[-1]>0:
            raise SystemExit('w2_l and w2_h should have different signs on bc_x1')
        if bcs[0]*bcs[1]<0:
            w2_h = w2_m
        else:
            w2_l = w2_m
        i += 1
        #print(bcs)
    sol = integrate.solve_ivp(osc_rot_odes, (x0, x1), y0, t_eval = t, args=(w2_m, Om2, n, theta, theta_p, x1, f_Om, gamma, A, fA, qA, fOm2), method=method, atol=1e-5)
    return sol, w2_m

def find_w2s(bcs, w2s):
    a = np.abs(bcs)
    ids = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    w2_temp = w2s[ids]
    if w2_temp[0] == w2s[0]:
        w2_temp = w2_temp[1:]
    if w2_temp[-1] == w2s[-1]:
        w2_temp = w2_temp[:-1]
    #print(w2_temp, bcs, w2s)
    return w2_temp



class modes:
    def __init__(self, p, w2s, n_modes=8, f_Om=0., gamma=5./3., tol_w2=1e-12):
        self.p = p
        self.w2s = w2s
        self.f_Om = f_Om
        self.gamma = gamma
        self.tol_w2 = tol_w2
        self.scan(p)
        self.find_modes(p, n_modes)
        
        
    def scan(self, p):
        print('Scanning...')
        bcs = np.zeros_like(self.w2s)
        #print(self.w2s)
        for i in tqdm(range(len(self.w2s))):
            x = self.w2s[i]
            #try:
            res = osc_rot_check_bc(w2=x, Om2=p.Om2, n=p.n, theta=p.theta, theta_p=p.theta_p, x0=1e-12, x1=p.x1, f_Om=self.f_Om, gamma=self.gamma, A=p.A, fA=p.fA, qA=p.qA, fOm2=p.fOm2)
            #except:
            #    res = np.nan
            bcs[i] = res
        self.bcs = bcs
        #print(self.bcs)
        self.w2_temp = find_w2s(bcs, self.w2s)
        #print(self.w2_temp)
        print('Scan completed!')
        return bcs
    
    def find_modes(self, p, n_modes):
        modes = []
        i = 1
        for x in self.w2_temp[:n_modes]:
            idx = self.w2s.tolist().index(x)
            w2_l = self.w2s[idx-1]
            w2_h = self.w2s[idx+1]
            sol, w2 = osc_rot_solutions(p.n, p.theta, p.theta_p, x0=1e-12, x1=p.x1, tol_b=1e-5, tol_w2=self.tol_w2, w2_l=w2_l, w2_h=w2_h, Om2=p.Om2, A=p.A, f_Om=self.f_Om, gamma=self.gamma, verbose=True, fA=p.fA, qA=p.qA, fOm2=p.fOm2)
            t = sol.t
            y = sol.y[0]
            t /= t[-1]
            y /= y[-1]
            
            md = {}
            md['om2'] = w2
            md['x'] = t
            md['zeta'] = y
            modes.append(md)
            #print(i, w2)
            i += 1
        self.modes = modes


def dw2_Om2_slope(m0, nn=10, structure_change=True):
    p0 = m0.p
    f_Om = m0.f_Om
    gamma = m0.gamma
    r = m0.modes[0]['x'][nn:-nn]
    z = m0.modes[0]['zeta'][nn:-nn]
    xi = r*p0.x1

    rho = p0.theta(xi)**(p0.n)
    dzdr = np.diff(z)/np.diff(r)
    dzdr = np.concatenate(([0.], dzdr))

    d2zdr2 = (z[2:]+z[:-2]-2*z[1:-1])/np.diff(r)[1:]**2
    d2zdr2 = np.concatenate((d2zdr2[:1], d2zdr2, d2zdr2[-1:]))

    theta1 = p0.theta1(xi)
    theta1_p = p0.theta1_p(xi)

    theta = p0.theta(xi)
    theta_p = p0.theta_p(xi)
    
    if not structure_change:
        fac = sum(rho*r**5*dzdr*z)/sum(rho*r**4*z*z) * gamma
        fac = -(2*f_Om + 3*gamma - 1) - fac
    else:
        l1 = -p0.theta_p(xi)/xi* ((3*gamma-4)*z + gamma*r*dzdr )
        l2 = -gamma/(p0.n+1)* p0.theta(xi)/xi**2* (4*r*dzdr + r**2*d2zdr2)
        drdm = -p0.theta1(p0.x1)/(p0.x1*p0.theta_p(p0.x1)) + p0.theta1(p0.x1)*p0.theta_pp(p0.x1)/p0.theta_p(p0.x1)**2 - p0.theta1_p(p0.x1)/p0.theta_p(p0.x1) + p0.x1/p0.theta_p(p0.x1) 
        drdm = -3*p0.theta1(p0.x1)/(p0.x1*p0.theta_p(p0.x1)) - p0.theta1_p(p0.x1)/p0.theta_p(p0.x1) + p0.x1/p0.theta_p(p0.x1) 
        dl = (theta1_p/theta_p + drdm) *l1 + (theta1/theta + drdm)*l2
        fac = sum(rho*r**4*dl*z)/sum(rho*r**4*z*z)
        fac = -(2*f_Om + 3) + fac
    return fac





"""
Solving for 4-parameter oscillation modes that include differential 
"""

# def osc_diff_rot_odes(x, y, w2, Om2, n, theta, theta_p, x1, gamma, A, q, nu):
#     res1 = -1/x*(3*y[0] + 1/gamma*y[1])

#     r3M = -theta_p(x1)/x1 + A
#     r3M /= -theta_p(x)/x + A
#     fac0 = 4 + (-1*Om2+w2) *r3M
#     fac1 = 1 - Om2 *r3M
#     fac2 = 2*Om2 *r3M
#     res2 = (1+n)*(-theta_p(x)+A*x)/theta(x)*(fac0*y[0]+fac1*y[1]+fac2*y[2])

#     res3 = 1/x* q*(-6*y[0] - 2/gamma*y[1] - y[2]+y[3])

#     res4 = -1j *np.sqrt(w2) *(x/x1)**2/q/nu *(2*y[0] + y[2])
#     return res1, res2, res3, res4


def osc_diff_rot_odes(x, y, w, Om2, n, theta, theta_p, x1, gamma, A, q, nu):
    w2 = w[0]**2 - w[1]**2 + 2*w[0]*w[1]*1j
    res1 = -1/x*(3*y[0] + 1/gamma*y[1])
    r3M = -theta_p(x1)/x1 + A
    r3M /= -theta_p(x)/x + A
    fac0 = 4 + (-1*Om2+w2) *r3M
    fac1 = 1 - Om2 *r3M
    fac2 = 2*Om2 *r3M
    res2 = (1+n)*(-theta_p(x)+A*x)/theta(x)*(fac0*y[0]+fac1*y[1]+fac2*y[2])
    res3 = 1/x* q*(-6*y[0] - 2/gamma*y[1] - y[2]+y[3])
    res4 = -1j *(w[0]+w[1]*1j) *(x/x1)**2/q/nu *(2*y[0] + y[2])
    return res1, res2, res3, res4







class osc_rad:
    def __init__(self, p, gamma=5./3., fixed_gamma=False):
        '''
        adiabatic radial perturbations
        p: poly_star or any class contains: V, c_1, x1, and any other required functions.
        Notes:
            - the inner/outer boundary conditions are supposed to locate at center/surface, make sure bvp_shooting fits this.
        '''
        self.p = p
        self.ndim = 2
        self.complex = False

        if fixed_gamma or not ('gamma' in p.__dict__):
            self.gamma = gamma
            self.fixed_gamma = True
        else:
            self.fixed_gamma = False

    def eqs(self, x, om):
        V = self.p.V(x)
        c_1 = self.p.c_1(x)
        if self.fixed_gamma:
            gamma = self.gamma
        else:
            gamma = self.p.gamma(x)

        eqs = np.zeros((self.ndim, self.ndim))
        eqs[0, 0] = -3.
        eqs[0, 1] = -1./gamma
        eqs[1, 0] = V *(4.+c_1*om**2)
        eqs[1, 1] = V

        return eqs

    def bds_in(self, om):
        if self.fixed_gamma:
            gamma = self.gamma
        else:
            gamma = self.p.gamma(0.)

        bds = np.zeros((1, self.ndim))
        bds[0, 0] = 3.
        bds[0, 1] = 1./gamma

        return bds

    def bds_out(self, om):

        bds = np.zeros((1, self.ndim))
        bds[0, 0] = 4. + om**2
        bds[0, 1] = 1.

        return bds




class osc_diff_rot_ad:
    def __init__(self, p, gamma=5./3., fixed_gamma=False):
        '''
        adiabatic perturbation with differential rotations
        p: poly_star or any class contains: V, c_1, x1, and any other required functions.
        Notes:
            - the inner/outer boundary conditions are supposed to locate at center/surface, make sure bvp_shooting fits this.
        '''
        self.p = p
        self.ndim = 4
        if fixed_gamma or not ('gamma' in p.__dict__):
            self.gamma = gamma
            self.fixed_gamma = True
        else:
            self.fixed_gamma = False
        self.complex = True

    def eqs(self, x, om):
        V = self.p.V(x)
        c_1 = self.p.c_1(x)
        Om2 = self.p.fOm2(x)
        q = self.p.q(x)
        nu = self.p.nu(x)
        if self.fixed_gamma:
            gamma = self.gamma
        else:
            gamma = self.p.gamma(x)
        x1 = self.p.x1

        eqs = np.zeros((self.ndim, self.ndim), dtype=np.complex_)
        eqs[0, 0] = -3.
        eqs[0, 1] = -1./gamma
        eqs[0, 2] = 0.
        eqs[0, 3] = 0.

        eqs[1, 0] = V *(4.+c_1*om**2-c_1*Om2)
        eqs[1, 1] = V *(1.-c_1*Om2)
        eqs[1, 2] = V * 2.*c_1*Om2
        eqs[1, 3] = 0.

        eqs[2, 0] = q *(-6.)
        eqs[2, 1] = q *(-2./gamma)
        eqs[2, 2] = q *(-1.)
        eqs[2, 3] = q *(1.)

        eqs[3, 0] = -1j*om* (x/x1)**2/q/nu *2
        eqs[3, 1] = 0.
        eqs[3, 2] = -1j*om* (x/x1)**2/q/nu
        eqs[3, 3] = 0.

        return eqs

    def bds_in(self, om):
        if self.fixed_gamma:
            gamma = self.gamma
        else:
            gamma = self.p.gamma(0.)

        bds = np.zeros((2, self.ndim), dtype=np.complex_)
        bds[0, 0] = 3.
        bds[0, 1] = 1./gamma
        bds[0, 2] = 0.
        bds[0, 3] = 0.

        bds[1, 0] = 2.
        bds[1, 1] = 0.
        bds[1, 2] = 1.
        bds[1, 3] = 0.

        return bds

    def bds_out(self, om):
        Om2 = self.p.fOm2(self.p.x1)

        bds = np.zeros((2, self.ndim), dtype=np.complex_)
        bds[0, 0] = 4. + om**2 -Om2
        bds[0, 1] = 1. - Om2
        bds[0, 2] = 2. * Om2
        bds[0, 3] = 0.

        bds[1, 0] = 2.
        bds[1, 1] = 0.
        bds[1, 2] = 1.
        bds[1, 3] = 0.

        return bds



def solve_homogeneous_linear_equations(U):
    '''
    https://stackoverflow.com/questions/1835246/how-to-solve-homogeneous-linear-equations-with-numpy
    '''
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
    # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(np.abs(e_vals))] 


def find_local_minima_2d(array2d):
    '''
    https://stackoverflow.com/questions/3986345/how-to-find-the-local-minima-of-a-smooth-multidimensional-array-in-numpy-efficie
    '''
    return ((array2d <= np.roll(array2d,  1, 0)) &
            (array2d <= np.roll(array2d, -1, 0)) &
            (array2d <= np.roll(array2d,  1, 1)) &
            (array2d <= np.roll(array2d, -1, 1)))


class bvp_shooting:
    def __init__(self, bvp):
        self.bvp = bvp

    def build(self, xs, om, debug=False, scheme='middle1'):
        '''
        build the n*N rank matrix
        xs[0] is always assumed to be 0.
        '''
        N = len(xs)
        ndim = self.bvp.ndim
        if not self.bvp.complex:
            dtype = float
        else:
            dtype = np.complex_
        S = np.zeros((ndim*N, ndim*N), dtype=dtype)

        bds_in = self.bvp.bds_in(om)
        i_min, j_min = 0, 0
        i_max, j_max = bds_in.shape
        S[i_min:i_max, j_min:j_max] = bds_in
        i_min, j_min = i_max, 0

        for i in range(1, N):
            if scheme == 'right':
                S[i_min:i_min+ndim, j_min:j_min+ndim] = -np.linalg.inv(np.identity(ndim) - (xs[i]-xs[i-1])/xs[i] * self.bvp.eqs(xs[i], om))
                S[i_min:i_min+ndim, j_min+ndim:j_min+ndim*2] = np.identity(ndim)
            elif scheme == 'middle1':
                T = (xs[i]-xs[i-1])/(xs[i]+xs[i-1])*self.bvp.eqs((xs[i]+xs[i-1])/2, om)
                S[i_min:i_min+ndim, j_min:j_min+ndim] = np.identity(ndim) +T
                S[i_min:i_min+ndim, j_min+ndim:j_min+ndim*2] = -np.identity(ndim) +T
            elif scheme == 'middle2':
                T = (xs[i]-xs[i-1])/(xs[i]+xs[i-1])*self.bvp.eqs((xs[i]+xs[i-1])/2, om)
                S[i_min:i_min+ndim, j_min:j_min+ndim] = -np.matmul(np.linalg.inv(np.identity(ndim)-T), np.identity(ndim)+T)
                S[i_min:i_min+ndim, j_min+ndim:j_min+ndim*2] = np.identity(ndim)
            i_min += ndim
            j_min += ndim

        bds_out = self.bvp.bds_out(om)
        di, dj = bds_out.shape
        S[i_min:i_min+di, j_min:j_min+dj] = bds_out

        if debug:
            if S.dtype==np.complex_:
                print(S.real)
                print(S.imag)
            else:
                print(S)
        
        return np.linalg.det(S), S

    def build_scan(self, xs, oms_real, oms_imag=None, scheme='middle1'):
        self.oms_real = oms_real
        self.oms_imag = oms_imag
        if not self.bvp.complex:
            detS = np.zeros_like(oms_real)
            for i in range(len(oms_real)):
                om = oms_real[i]
                detS[i], _ = self.build(xs, om=om, scheme=scheme)
            self.oms = oms_real
            self.detS = detS
            return oms_real, detS

        else:
            oms = np.zeros((oms_real.shape[0], oms_imag.shape[0]), dtype=np.complex_)
            detS = np.zeros_like(oms, dtype=np.complex_)
            for i in tqdm(range(oms.shape[0])):
                for j in range(oms.shape[1]):
                    om = oms_real[i] + 1j* oms_imag[j]
                    oms[i, j] = om
                    detS[i, j], _ = self.build(xs, om=om, scheme=scheme)
            self.oms = oms
            self.detS = detS
            return oms, detS

    
    def plot_abs_contours(self, save_path=''):
        '''
        Plot norm of the determinant (real, imag, abs)
        '''
        oms_real, oms_imag = self.oms_real, self.oms_imag
        dx = np.diff(oms_real)[0]
        dy = np.diff(oms_imag)[0]
        extent = [oms_real.min()-dx/2, oms_real.max()+dx/2, oms_imag.min()-dy/2, oms_imag.max()+dy/2]

        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        ax = axs[0]
        t = self.detS.real.T
        ax.imshow(np.log10(np.abs(t)), alpha=1, extent=extent, origin='lower')
        ax.set_xlabel(r'${\rm Re~}\tilde{\omega}$')
        ax.set_ylabel(r'${\rm Im~}\tilde{\omega}$')
        ax.set_title(r'$\lg(|{\rm Re~}\mathcal{D}(\tilde{\omega})|)$')

        ax = axs[1]
        t = self.detS.imag.T
        ax.imshow(np.log10(np.abs(t)), alpha=1, extent=extent, origin='lower')
        ax.set_xlabel(r'${\rm Re~}\tilde{\omega}$')
        ax.set_ylabel(r'${\rm Im~}\tilde{\omega}$')
        ax.set_title(r'$\lg(|{\rm Im~}\mathcal{D}(\tilde{\omega})|)$')

        ax = axs[2]
        ax.imshow(np.log10(np.absolute(self.detS.T)), alpha=2, extent=extent, origin='lower')
        ax.set_xlim((None, extent[1]))
        ax.set_xlabel(r'${\rm Re~}\tilde{\omega}$')
        ax.set_ylabel(r'${\rm Im~}\tilde{\omega}$')
        ax.set_title(r'$\lg(|\mathcal{D}(\tilde{\omega})|)$')

        if save_path != '':
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        # return fig


    def plot_zero_contours(self, save_path=''):
        '''
        Plot zero contours of the determinant (real, imag)
        '''
        fig, ax = plt.subplots()
        X, Y = np.meshgrid(self.oms_real, self.oms_imag)
        Z = self.detS.real.T
        ax.contour(X, Y, Z, [0.], colors='r')
        ax.contourf(X, Y, Z, [0., np.inf], colors='r', alpha=0.2)

        Z = self.detS.imag.T
        ax.contour(X, Y, Z, [0.], colors='k')
        ax.contourf(X, Y, Z, [0., np.inf], colors='k', alpha=0.2)

        ax.set_aspect('equal')
        ax.set_xlabel(r'${\rm Re~}\tilde{\omega}$')
        ax.set_ylabel(r'${\rm Im~}\tilde{\omega}$')
        if save_path != '':
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        # return fig


    def set_initial_brackets(self, d=3):
        if not self.bvp.complex:
            oms = self.oms
            detS = np.abs(self.detS)
            a = detS
            ids = np.arange(len(a))[np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True] ]
            if ids[0]<1:
                ids = ids[1:]
            if ids[-1]>=len(oms)-1:
                ids = ids[:-1]
            oms_l = oms[ids-1]
            oms_h = oms[ids+1]
            return np.transpose([oms_l, oms_h])
        else:
            oms = self.oms
            detS = self.detS
            nx, ny = oms.shape
            mesh = np.array([[(i, j) for j in range(ny)] for i in range(nx)])
            crit = find_local_minima_2d(np.absolute(detS))
            ims = mesh[crit]
            pts_center_temp = oms[crit]
            # local minimum: initial guess; then we try to find the four neighboring points
            pts_center = []
            pts_neighbors = []

            for k in range(len(ims)):
                im = ims[k]
                pt_center = pts_center_temp[k]
                ims_temp = [[(i>im[0]-d) & (i<im[0]+d+1) & (j>im[1]-d) & (j<im[1]+d+1) for j in range(ny)] for i in range(nx)]
                pts = []
                sucess = True
                for ij in [[1, 1], [1, -1], [-1, -1], [-1, 1]]:
                    i = ij[0]
                    j = ij[1]
                    crit = ims_temp & (i*detS.real>0) & (j*detS.imag>0)
                    if not crit.any():
                        # check if the point is valid
                        sucess = False
                        warnings.warn("Invalid guess at %f + %f i"%(pt_center.real, pt_center.imag))
                        break
                    pts_temp = oms[crit]
                    pts.append(pts_temp[np.absolute(pts_temp - pt_center).argmin()])
                if not sucess:
                    continue
                else:
                    pts_center.append(pt_center)
                    pts_neighbors.append(pts)

            return np.array(pts_center), np.array(pts_neighbors)

    def find_eigenvalues(self, xs, method='Nelder-Mead', options={'xatol':1e-6}, loss_function='abs', xtol=1e-12, debug=False, iter_max=100, d=2, Dtol=1e-9):
        '''
        Method:
            - Zigzag (my method)
            - methods in optimize, e.g., Nelder-Mead
        '''
        eigenvalues = []
        if not self.bvp.complex:
            brackets = self.set_initial_brackets()
            D = lambda om: self.build(xs, om)[0]
            for bracket in brackets:
                eigenvalues.append(optimize.ridder(D, *bracket))
        else:
            oms, brackets = self.set_initial_brackets(d=d)
            if loss_function == 'abs':
                D = lambda om: np.absolute(self.build(xs, om[0]+om[1]*1j)[0])
            elif loss_function == 'multiply':
                def D(om):
                    detS, _ = self.build(xs, om[0]+om[1]*1j)
                    return np.abs(detS.real * detS.imag)
            
            for i in range(len(oms)):
                om = oms[i]
                pts = brackets[i]
                if method == 'Zigzag':
                    eps = 1e100
                    epsD = 1e100
                    k = 0
                    while (eps>xtol or epsD>Dtol) and k<iter_max:
                        # move
                        for i in range(len(pts)):
                            pair = pts[[i, (i+1)%len(pts)]]
                            mid = (pair[0]+pair[1])/2
                            detS_m, _ = self.build(xs, mid)
                            detS_p, _ = self.build(xs, pair[0])
                            # if debug:
                            #     print(pts)
                            if np.sign(detS_m.real)*np.sign(detS_p.real)>0 and np.sign(detS_m.imag)*np.sign(detS_p.imag)>0:
                                # shrink one edge
                                pts[i] = mid
                            else:
                                pts[(i+1)%(len(pts))] = mid
                        mid = np.mean(pts)
                        eps = max(np.absolute(pts - mid))
                        epsD = np.abs(detS_m)
                        # this is probably not a very good idea
                        k += 1
                        if debug:
                            print(k, mid, eps, epsD)
                    if eps>xtol:
                        warnings.warn("Eigenvalue tolerance target failed at %f + %f i, tol=%e"%(om.real, om.imag, eps))
                    if epsD>Dtol:
                        warnings.warn("Determinant tolerance target failed at %f + %f i, tol=%e"%(om.real, om.imag, epsD))
                    if eps>max(1e-4, xtol):
                        # there may still be some invalid guess.
                        continue
                    eigenvalues.append(mid)
                else:
                    res = optimize.minimize(D, [om.real, om.imag], method=method, options=options)
                    if res.success:
                        eigenvalues.append(res.x[0]+1j*res.x[1])
                    else:
                        warnings.warn("Local mimima optimization failed at %f + %f i"%(om.real, om.imag))
        self.eigenvalues = eigenvalues
        return eigenvalues

    def find_eigenfunctions(self, xs, normalize=True, debug=False):
        eigenfunctions = []
        for eigenvalue in self.eigenvalues:
            _, S = self.build(xs, eigenvalue)
            v = solve_homogeneous_linear_equations(S)
            if normalize:
                v /= v[len(S)-self.bvp.ndim]
            if debug:
                print(eigenvalue, max(np.absolute(np.dot(S, v.T))))
            v = v.reshape((len(v)//self.bvp.ndim, self.bvp.ndim)).T
            eigenfunctions.append(v)
        self.eigenfunctions = eigenfunctions
        return eigenfunctions
            
            
class mesa_star:
    '''
    read mesa file for gyre and construct a star model
    '''
    def __init__(self, file):
        with open(file, 'r') as ff:
            rs = ff.readline().split()
            ver = int(rs[-1])
            M = float(rs[1])
            R = float(rs[2])
            Omc = np.sqrt(const_G*M/R**3)
        self.ver = ver
        data = np.loadtxt(file, skiprows=1)
        xs = data[:,1]
        self.xs = xs
        self.x1 = R

        Mrs = data[:,2]
        c1s = (xs/R)**3 * (M/Mrs)
        self.c_1 = interpolate.interp1d(xs, c1s, fill_value='extrapolate')

        gs = const_G *Mrs / (xs)**2
        ps = data[:,4]
        rhos = data[:,6]
        Vs = rhos*gs*xs/ps
        if not np.isfinite(Vs[0]):
            Vs[0] = Vs[1]
        self.V = interpolate.interp1d(xs, Vs, fill_value='extrapolate')

        N2s = data[:,8]
        gammas = data[:,9]
        self.gamma = interpolate.interp1d(xs, gammas, fill_value='extrapolate')

        Om2s = (data[:,18]/Omc)**2
        self.fOm2 = interpolate.interp1d(xs, Om2s, fill_value='extrapolate')
        if ver == 233:
            qs = data[:,19]
            self.q = interpolate.interp1d(xs, qs, fill_value='extrapolate')

            nus = data[:,21]/np.sqrt(const_G*M*R)
            self.nu = interpolate.interp1d(xs, nus, fill_value='extrapolate')


