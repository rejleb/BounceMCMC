import state
import mcmc
import observations
import driver
import ttvfast
import scipy.stats
import copy
import numpy as np
from datetime import datetime
from scipy import stats

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
        self.Nvars = 21
        
    def get_logp(self, obs):
        if (self.priorHard()):
            lnpri = -np.inf
            return lnpri
        softlnpri = 0.0
        if self.logp is None:
            self.logp = -self.get_chi2(obs)
        return self.logp + softlnpri
        
    def get_chi2(self, obs):
        params = self.stel_m_planets
        planet1 = ttvfast.models.Planet(*params[:7])
        planet2 = ttvfast.models.Planet(*params[7:14])
        planet3 = ttvfast.models.Planet(*params[14:])
        stellar_mass = 0.74
        planets = [planet1, planet2, planet3]            
        #get results...
        results = ttvfast.ttvfast(planets, stellar_mass, self.Time, self.dt, self.Total)
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
            #print "Number of transits did not match [{a}=/={b}], logp=inf.".format(a=count,b=len(obs.times),time=datetime.utcnow())
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
    
    def get_ttv(self, obs,transitcheck=True):
        params = self.stel_m_planets
        planet1 = ttvfast.models.Planet(*params[:7])
        planet2 = ttvfast.models.Planet(*params[7:14])
        planet3 = ttvfast.models.Planet(*params[14:])
        stellar_mass = 0.74
        planets = [planet1, planet2, planet3]            
        results = ttvfast.ttvfast(planets, stellar_mass, self.Time, self.dt, self.Total)
        integer_indices, epochs, times, rsky, vsky = results["positions"]
        t1_times = []
        t2_times = []
        t3_times = []
        count = 0
        for i in range(len(times)):
            if(integer_indices[i]==0):
                if( not np.isclose(times[i],-2.0)):
                    t1_times.append(times[i])
                    count +=1
                else:
                    break
            elif(integer_indices[i]==1):
                if( not np.isclose(times[i],-2.0)):
                    t2_times.append(times[i])
                else:
                    break
            else:
                t3_times.append(times[i])
        if(transitcheck):
            if(count != len(obs.times)):
                print "Number of transits did not match [{a}=/={b}], logp=inf.".format(a=count,b=len(obs.times),time=datetime.utcnow())
                print t1_times
                print t2_times
                print t3_times
                return np.inf
        print t1_times
        print t2_times
        print t3_times
        return t1_times

    def get_ttv_plotting(self, obs, checktransits=True):
        t1_times = self.get_ttv(obs, checktransits)
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(t1_times)),t1_times)
        diff = (np.arange(len(t1_times))*slope + intercept) - np.asarray(t1_times)
        return t1_times, diff

    #Will need to modify this...
    def deepcopy(self):
        return TTVState(copy.deepcopy(self.stel_m_planets), copy.deepcopy(self.ttvfast_settings))

    def priorHard(self):
        for i in enumerate(self.stel_m_planets):
            if (self.stel_m_planets[0] <= 1e-10) :
                #print "Invalid state was proposed (m1)"
                return True
        return False

