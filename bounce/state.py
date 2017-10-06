import numpy as np
import rebound
import copy
from datetime import datetime
import matplotlib.pyplot as plt

class State(object):
    def __init__(self, planets, ignore_vars=[], ignore_params=None):
        self.planets = planets
        self.logp = None
        self.logp_d = None
        self.logp_dd = None
        self.logp_dd_lam = None
        self.Ginv = None
        self.Ginvsqrt = None
        self.planets_vars = []
        self.Nvars = 0
        self.hillRadiusMax = 0.0
        self.hillRadiusFactor = 1.
        self.collisionGhostParams = None
        #These variables are used to track the planets in troubleshooting. Kept here for convenience. 
        self.planet1x = []
        self.planet1y = []
        self.planet2x = []
        self.planet2y = []

        self.ignore_vars = ignore_vars
        self.ignore_params = ignore_params
        for p, planet in enumerate(planets):
            planet_vars = [x for x in planet.keys() if (x not in ignore_vars)]
            if(ignore_params != None):
                for o in range(len(ignore_params[p])):
                    planet_vars.remove(ignore_params[p][o])
            self.planets_vars.append(planet_vars)
            self.Nvars += len(planet_vars)

    '''
    This is used to setup a simulation to compute the logp. Sets the collision distance to a 'factor' of the largest hill radius among planets.
    '''
    def setup_sim(self):
        sim = rebound.Simulation()
        sim.add(m=1.)
        self.hillRadiusMax = 0
        for planet in self.planets:
            sim.add(primary=sim.particles[0],**planet)
            r = planet['a']*(planet['m']/(3*sim.particles[0].m))**(1./3.)
            if( r > self.hillRadiusMax):
                self.hillRadiusMax = r
        sim.move_to_com()
        sim.exit_min_distance = self.hillRadiusFactor * self.hillRadiusMax
        return sim

    '''
    This function is called when troubleshooting. The following statement are called between integration steps.
    '''
    def check_ts(self, sim):
        self.planet1x.append(sim.contents.particles[1].x)
        self.planet1y.append(sim.contents.particles[1].y)
        self.planet2x.append(sim.contents.particles[2].x)
        self.planet2y.append(sim.contents.particles[2].y)
           
    '''
    This returns the rv of the system given an array of times. The troubleshooting variables+check_ts are initialized and ran here.
    '''
    def get_rv(self, times):
        sim = self.setup_sim()
        rv = np.zeros(len(times))
        #self.planet1x = []
        #self.planet1y = []
        #self.planet2x = []
        #self.planet2y = []
        for i, t in enumerate(times):
            #This is used to check the position of a planet inbetween timesteps
            #sim.post_timestep_modifications = self.check_ts
            sim.integrate(t)
            rv[i] = sim.particles[0].vx
        return rv

    '''
    This returns the rv of the system but rather than the times of the obs, the time array used is an interpolation of 1000 points between the extremum of the data. Used for plotting.
    '''
    def get_rv_plotting(self, obs, Npoints=1000):
        times = np.linspace(obs.tb[0],obs.tf[len(obs.tf)-1],Npoints)
        try:
            a = self.get_rv(times)
        except rebound.Encounter as e:
            print "You are trying to plot a set parameters which give a collision."
        return times, a

    '''
    This returns the chi2 value of a state given a set of obs.
    '''
    def get_chi2(self, obs):
        rvf = self.get_rv(obs.tf)
        rvb = self.get_rv(obs.tb)        
        chi2f = 0.
        chi2b = 0.
        for i, tf in enumerate(obs.tf):
            chi2f += ((rvf[i]-obs.rvf[i])**2.)/(obs.errorf[i]**2.)
        for i, tb in enumerate(obs.tb):
            chi2b += ((rvb[i]-obs.rvb[i])**2.)/(obs.errorb[i]**2.)
        return (chi2b+chi2f)/(obs.Npoints)

    '''
    Returns the simple logp of the state.
    '''
    def get_logp(self, obs):
        if (self.priorHard()):
            lnpri = -np.inf
            return lnpri
        softlnpri = 0.0
        if self.logp is None:
            self.logp = -self.get_chi2(obs)
        return self.logp + softlnpri
    
    '''
    This is the lnprior function used with the emcee package and can be passed as an argument. May be moved to MCMC class in later updates.
    '''
    def lnprior(theta):
        m, a, h, k, l = theta
        if (1e-7 < m < 0.1) and (1e-2 < a < 500.0) and ((h**2 + k**2) < 1.0) and (-2*np.pi < l < 2*np.pi):
            return 0.0
        return -np.inf

    '''
    Used in MH algorithm to generate proposals.
    '''
    def shift_params(self, vec):
        self.logp = None
        if len(vec)!=self.Nvars:
            raise AttributeError("vector has wrong length")
        varindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if k not in self.ignore_vars:
                    if(self.ignore_params != None):
                        if (k not in self.ignore_vars) and (k not in self.ignore_params[i]):
                            self.planets[i][k] += vec[varindex]
                            varindex += 1
                    elif(k not in self.ignore_vars):
                        self.planets[i][k] += vec[varindex]
                        varindex += 1
   
    '''
    Returns the parameters of a state.
    '''
    def get_params(self):
        params = np.zeros(self.Nvars)
        parindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if(self.ignore_params != None):
                    if (k not in self.ignore_vars) and (k not in self.ignore_params[i]):
                        params[parindex] = self.planets[i][k]
                        parindex += 1
                elif(k not in self.ignore_vars):
                    params[parindex] = self.planets[i][k]
                    parindex += 1
        return params

    '''
    Sets the parameters of a state.
    '''
    def set_params(self, vec):
        self.logp = None
        if len(vec)!=self.Nvars:
            raise AttributeError("vector has wrong length")
        varindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if(self.ignore_params != None):
                    if (k not in self.ignore_vars) and (k not in self.ignore_params[i]):
                        self.planets[i][k] = vec[varindex]
                        varindex += 1
                elif(k not in self.ignore_vars):
                    self.planets[i][k] = vec[varindex]
                    varindex += 1
    
    '''
    Returns the keys/labels of parameters.
    '''
    def get_keys(self):
        keys = [""]*self.Nvars
        parindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if(self.ignore_params != None):
                    if (k not in self.ignore_vars) and (k not in self.ignore_params[i]):
                        keys[parindex] = "$%s_%d$"%(k,i)
                        parindex += 1
                elif(k not in self.ignore_vars):
                    keys[parindex] = "$%s_%d$"%(k,i)
                    parindex += 1
        return keys

    '''
    Returns the keys/labels without type conversion.
    '''
    def get_rawkeys(self):
        keys = [""]*self.Nvars
        parindex = 0
        for i, planet in enumerate(self.planets):
            for k in planet.keys():
                if(self.ignore_params != None):
                    if (k not in self.ignore_vars) and (k not in self.ignore_params[i]):
                        keys[parindex] = k
                        parindex += 1
                elif(k not in self.ignore_vars):
                    keys[parindex] = k
                    parindex += 1
        return keys

    '''
    Returns a deepcopy of current state.
    '''
    def deepcopy(self):
        return State(copy.deepcopy(self.planets), copy.deepcopy(self.ignore_vars), ignore_params=copy.deepcopy(self.ignore_params))

    '''
    Helps manage the variational index in the simulation setup.
    '''
    def var_pindex_vname(self, vindex):
        vi = 0.
        for pindex, p in enumerate(self.planets_vars):
            for v in p:
                if vindex == vi:
                    return pindex+1, v
                vi += 1

    '''
    Sets up the a simulation with variations in pal coordinates.
    '''
    def setup_sim_vars(self):
        sim = self.setup_sim()
        variations1 = []
        variations2 = []
        for vindex in range(self.Nvars):
            pindex, vname = self.var_pindex_vname(vindex)
            v = sim.add_variation(order=1)
            v.vary(pindex,vname)
            variations1.append(v)
        for vindex1 in range(self.Nvars):
            for vindex2 in range(self.Nvars):
                if vindex1 >= vindex2:
                    pindex1, vname1 = self.var_pindex_vname(vindex1)
                    pindex2, vname2 = self.var_pindex_vname(vindex2)
                    v = sim.add_variation(order=2, first_order=variations1[vindex1], first_order_2=variations1[vindex2])
                    if pindex1 == pindex2:
                        v.vary(pindex1,vname1,vname2)
                    variations2.append(v)
        sim.move_to_com()
        return sim, variations1, variations2

    '''
    Sets up the a simulation with variations in pal coordinates.
    '''
    def setup_sim_vars_grad(self):
        sim = self.setup_sim()
        variations1 = []
        for vindex in range(self.Nvars):
            pindex, vname = self.var_pindex_vname(vindex)
            v = sim.add_variation(order=1)
            v.vary(pindex,vname)
            variations1.append(v)
        sim.move_to_com()
        return sim, variations1

    '''
    Gets the chi2, chi2_d, chi2_dd values for the current state.
    '''
    def get_chi2_d_dd(self, obs):
        sim, variations1, variations2 = self.setup_sim_vars()
        chi2b = 0.
        chi2_db = np.zeros(self.Nvars)
        chi2_ddb = np.zeros((self.Nvars,self.Nvars))
        chi2f = 0.
        chi2_df = np.zeros(self.Nvars)
        chi2_ddf = np.zeros((self.Nvars,self.Nvars))
        fac = obs.Npoints
        for i, tf in enumerate(obs.tf):
            sim.integrate(tf)
            chi2f += (sim.particles[0].vx-obs.rvf[i])**2*1./(obs.errorf[i]**2 *fac)
            v2index = 0
            for vindex1 in range(self.Nvars):
                chi2_df[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-obs.rvf[i])*1./(obs.errorf[i]**2 *fac)
                for vindex2 in range(self.Nvars):
                    if vindex1 >= vindex2:
                        chi2_ddf[vindex1][vindex2] +=  2. * variations2[v2index].particles[0].vx * (sim.particles[0].vx-obs.rvf[i])*1./(obs.errorf[i]**2 * fac) + 2. * variations1[vindex1].particles[0].vx * variations1[vindex2].particles[0].vx*1./(obs.errorf[i]**2 * fac)
                        v2index += 1
                        chi2_ddf[vindex2][vindex1] = chi2_ddf[vindex1][vindex2]
        sim, variations1, variations2 = self.setup_sim_vars()
        for i, tb in enumerate(list(reversed(obs.tb))):
            sim.integrate(tb)
            chi2b += (sim.particles[0].vx-list(reversed(obs.rvb))[i])**2*1./(list(reversed(obs.errorb))[i]**2 * fac)
            v2index = 0
            for vindex1 in range(self.Nvars):
                chi2_db[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-list(reversed(obs.rvb))[i])*1./(list(reversed(obs.errorb))[i]**2 * fac)
                for vindex2 in range(self.Nvars):
                    if vindex1 >= vindex2:
                        chi2_ddb[vindex1][vindex2] +=  2. * variations2[v2index].particles[0].vx * (sim.particles[0].vx-list(reversed(obs.rvb))[i])*1./(list(reversed(obs.errorb))[i]**2 * fac) + 2. * variations1[vindex1].particles[0].vx * variations1[vindex2].particles[0].vx*1./(list(reversed(obs.errorb))[i]**2 * fac)
                        v2index += 1
                        chi2_ddb[vindex2][vindex1] = chi2_ddb[vindex1][vindex2]
        return chi2b+chi2f, chi2_db+chi2_df, chi2_ddb+chi2_ddf

    def get_chi2_d_dd_analytical(self, obs):
        sim = self.setup_sim()
        chi2b = 0.
        chi2_db = np.zeros(self.Nvars)
        chi2_ddb = np.zeros((self.Nvars,self.Nvars))
        chi2f = 0.
        chi2_df = np.zeros(self.Nvars)
        chi2_ddf = np.zeros((self.Nvars,self.Nvars))
        fac = obs.Npoints
        
        for i, tf in enumerate(obs.tf):
            sim.integrate(tf)
            chi2f += (sim.particles[0].vx-obs.rvf[i])**2*1./(obs.errorf[i]**2 *fac)
            v2index = 0
            for vindex1 in range(self.Nvars):
                #chi2_df[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-obs.rvf[i])*1./(obs.errorf[i]**2 *fac)
                for vindex2 in range(self.Nvars):
                    if vindex1 >= vindex2:
                        #chi2_ddf[vindex1][vindex2] +=  2. * variations2[v2index].particles[0].vx * (sim.particles[0].vx-obs.rvf[i])*1./(obs.errorf[i]**2 * fac) + 2. * variations1[vindex1].particles[0].vx * variations1[vindex2].particles[0].vx*1./(obs.errorf[i]**2 * fac)
                        v2index += 1
                        chi2_ddf[vindex2][vindex1] = chi2_ddf[vindex1][vindex2]
        
        sim = self.setup_sim()
        for i, tb in enumerate(list(reversed(obs.tb))):
            sim.integrate(tb)
            chi2b += (sim.particles[0].vx-list(reversed(obs.rvb))[i])**2*1./(list(reversed(obs.errorb))[i]**2 * fac)
            v2index = 0
            for vindex1 in range(self.Nvars):
                #chi2_db[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-list(reversed(obs.rvb))[i])*1./(list(reversed(obs.errorb))[i]**2 * fac)
                for vindex2 in range(self.Nvars):
                    if vindex1 >= vindex2:
                        #chi2_ddb[vindex1][vindex2] +=  2. * variations2[v2index].particles[0].vx * (sim.particles[0].vx-list(reversed(obs.rvb))[i])*1./(list(reversed(obs.errorb))[i]**2 * fac) + 2. * variations1[vindex1].particles[0].vx * variations1[vindex2].particles[0].vx*1./(list(reversed(obs.errorb))[i]**2 * fac)
                        v2index += 1
                        chi2_ddb[vindex2][vindex1] = chi2_ddb[vindex1][vindex2]
        return chi2b+chi2f, chi2_db+chi2_df, chi2_ddb+chi2_ddf

    '''
    Gets the chi2, chi2_d values for the current state.
    '''
    def get_chi2_d(self, obs):
        sim, variations1 = self.setup_sim_vars_grad()
        chi2b = 0.
        chi2_db = np.zeros(self.Nvars)
        chi2f = 0.
        chi2_df = np.zeros(self.Nvars)
        fac = obs.Npoints
        for i, tf in enumerate(obs.tf):
            sim.integrate(tf)
            chi2f += (sim.particles[0].vx-obs.rvf[i])**2*1./(obs.errorf[i]**2 *fac)
            for vindex1 in range(self.Nvars):
                chi2_df[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-obs.rvf[i])*1./(obs.errorf[i]**2 *fac)
        sim, variations1 = self.setup_sim_vars_grad()
        for i, tb in enumerate(list(reversed(obs.tb))):
            sim.integrate(tb)
            chi2b += (sim.particles[0].vx-list(reversed(obs.rvb))[i])**2*1./(list(reversed(obs.errorb))[i]**2 * fac)
            for vindex1 in range(self.Nvars):
                chi2_db[vindex1] += 2. * variations1[vindex1].particles[0].vx * (sim.particles[0].vx-list(reversed(obs.rvb))[i])*1./(list(reversed(obs.errorb))[i]**2 * fac)
        return chi2b+chi2f, chi2_db+chi2_df

    '''
    Gets the logp_d_dd values using the chi2 function.
    '''
    def get_logp_d_dd(self, obs):
        if (self.logp is None) or (self.logp_d is None) or (self.logp_dd is None):
            chi, chi_d, chi_dd = self.get_chi2_d_dd(obs)
            self.logp, self.logp_d, self.logp_dd = -chi, -chi_d, -chi_dd
        return self.logp, self.logp_d, self.logp_dd

    def get_logp_d_dd_analytical(self, obs):
        if (self.logp is None) or (self.logp_d is None) or (self.logp_dd is None):
            chi, chi_d, chi_dd = self.get_chi2_d_dd_analytical(obs)
            self.logp, self.logp_d, self.logp_dd = -chi, -chi_d, -chi_dd
        return self.logp, self.logp_d, self.logp_dd

    '''
    Gets the logp_d values using the chi2 function.
    '''
    def get_logp_d(self, obs):
        if (self.logp is None) or (self.logp_d is None):
            chi, chi_d = self.get_chi2_d(obs)
            self.logp, self.logp_d = -chi, -chi_d
        return self.logp, self.logp_d

    '''
    This function behaves as a basic uniform prior. Gives out warning messages which may be useful in telling if an MCMC is struggling. Hardcodded values have been chosen as physical limits.
    '''
    def priorHard(self):
        for i, planet in enumerate(self.planets):
            if (self.planets[i]["a"] <=0.002) :
                print "Invalid state was proposed (a)"
                return True
            if (self.planets[i]["m"] <=1.e-8) :
                print "Invalid state was proposed (m)"
                return True
            if "h" in (self.planets[i].keys()) or ("k" in self.planets[i].keys()):
                if (self.planets[i]["h"]**2 + self.planets[i]["k"]**2 >=1.0) :
                    print "Invalid state was proposed (h & k)"
                    return True
            if "ix" in (self.planets[i].keys()) or ("iy" in self.planets[i].keys()):
                if (self.planets[i]["ix"]**2 + self.planets[i]["iy"]**2 >=4.0) :
                    print "Invalid state was proposed (ix & iy)"
                    return True
        return False