import state
import mcmc
import observations
import driver
import ttvfast
import scipy.stats
import copy
import numpy as np

class TTVState(object):
    def __init__(self, stel_m_planets, ttvfast_settings):
        self.stel_m_planets = stel_m_planets
        self.ttvfast_settings = ttvfast_settings
        self.Time = ttvfast_settings[0]
        self.dt = ttvfast_settings[1]
        self.Total = ttvfast_settings[2]
        self.n_plan = ttvfast_settings[3]
        self.input_flags = ttvfast_settings[4]
        self.logp = None
        self.Nvars = 14
        
    def get_logp(self, obs):
        if (self.priorHard()):
            lnpri = -np.inf
            return lnpri
        softlnpri = 0.0
        if self.logp is None:
            self.logp = -self.get_chi2(obs)
        return self.logp + softlnpri
        
    def get_chi2(self, obs):
        #split and prep data for ttvfast
        params = self.stel_m_planets
        #planet1 = ttvfast.models.Planet(*params[1:1 + 7])
        planet1 = ttvfast.models.Planet(*params[:7])
        #planet2 = ttvfast.models.Planet(*params[1 + 7:])
        planet2 = ttvfast.models.Planet(*params[7:])
        #stellar_mass = params[0]
        stellar_mass = 0.95573417954
        planets = [planet1, planet2]            
        #get results...
        results = ttvfast.ttvfast(planets, stellar_mass, self.Time, self.dt, self.Total)
        #prep results
        integer_indices, epochs, times, rsky, vsky = results["positions"]
        t1_times = []
        t2_times = []
        count = 0
        for i in range(len(times)):
            if(integer_indices[i]==0):
                if( not np.isclose(times[i],-2.0)):
                    t1_times.append(times[i])
                    count +=1
                else:
                    break
            else:
                t2_times.append(times[i])
        if(count != len(obs.times)):
            print count
            print len(obs.times)
            print "Number of transits did not match, logp=inf."
            return np.inf
        #calc chi2
        chi2 = 0.
        fac = len(t1_times)
        for i in range(len(t1_times)):
            chi2 += (t1_times[i] - obs.times[i])**2. * 1./(obs.errors[i])**2.
        return chi2

    def get_params(self):
        params = np.copy(self.stel_m_planets)
        return params
        
    def set_params(self, stel_m_planets):
        self.logp = None
        if len(stel_m_planets)!=self.Nvars:
            raise AttributeError("vector has wrong length")
        self.stel_m_planets = stel_m_planets
    
    #Will need to modify this...
    def deepcopy(self):
        return TTVState(copy.deepcopy(self.stel_m_planets), copy.deepcopy(self.ttvfast_settings))

    def priorHard(self):
        for i in enumerate(self.stel_m_planets):
            if (self.stel_m_planets[0] <= 0.0000001) :
                print "Invalid state was proposed (m1)"
                return True
        return False

