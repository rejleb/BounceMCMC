import numpy as np
import emcee
from scipy import stats
import rebound
from datetime import datetime
import sys
import traceback
import copy

'''
Parent MCMC class
'''
class Mcmc(object):
    def __init__(self, initial_state, obs):
        self.state = initial_state.deepcopy()
        self.obs = obs

    def step(self):
        return True 
    
    def step_force(self):
        tries = 1
        while self.step()==False:
            tries += 1
            pass
        return tries

#create a static lnprob function to pass to the emcee package
def lnprob(x, e):
    e.state.set_params(x)
    try:
        logp = e.state.get_logp(e.obs)
    except:
        print "Collision! {t}".format(t=datetime.utcnow())
        e.state.collisionGhostParams.append(e.state.deepcopy())
        #e.state.collisionGhostParams.append(e.state.get_params())
        #print e.state.collisionGhostParams
        return -np.inf
    return logp

'''
emcee MCMC coupled with rebound.
'''
class Ensemble(Mcmc):
    def __init__(self, initial_state, obs, scales, nwalkers=10, a=2, listmode=False):
        super(Ensemble,self).__init__(initial_state, obs)
        if(listmode):
            self.set_scales_listmode(scales)
        else:
            self.set_scales(scales)
        self.nwalkers = nwalkers
        self.states = [self.state.get_params() for i in range(nwalkers)]
        self.previous_states = [self.state.get_params() for i in range(nwalkers)]
        self.lnprob = None
        self.totalErrorCount = 0
        for i,s in enumerate(self.states):
            shift = 0.1e-2*self.scales*np.random.normal(size=self.state.Nvars)
            self.states[i] += shift
            self.previous_states[i] += shift
        self.sampler = emcee.EnsembleSampler(nwalkers,self.state.Nvars, lnprob, a=a, args=[self])

    '''
    Constitutes 1 emcee step.
    '''
    def step(self):
        self.previous_states = copy.deepcopy(self.states)
        self.state.collisionGhostParams = []
        self.states, self.lnprob, rstate = self.sampler.run_mcmc(self.states,1,lnprob0=self.lnprob)
        for i in range(len(self.states)):
            for j in range(len(self.states[0])):
                if(self.previous_states[i][j] != self.states[i][j]):
                    return True
        else:
            return False

    '''
    Sets the scales for the initial random distribution of walkers. Mileage may vary.
    '''
    def set_scales(self, scales):
        self.scales = np.ones(self.state.Nvars)
        keys = self.state.get_rawkeys()
        for i,k in enumerate(keys):
            if k in scales:
                self.scales[i] = scales[k]

    '''
    Sets the scales for the initial random distribution of walkers. Mileage may vary.
    '''
    def set_scales_listmode(self, scales):
        self.scales = scales

'''
Metropolis-Hastings MCMC coupled with rebound.
'''
class Mh(Mcmc):
    def __init__(self, initial_state, obs):
        super(Mh,self).__init__(initial_state, obs)
        #default value of 3e-5 for MH mcmc
        self.step_size = 3e-5

    '''
    Generates a proposal randomly.
    '''
    def generate_proposal(self):
        prop = self.state.deepcopy()
        shift = self.step_size*self.scales*np.random.normal(size=self.state.Nvars)
        prop.shift_params(shift)
        return prop

    '''
    Sets the scales used in MH's random steps.
    '''
    def set_scales(self, scales):
        self.scales = np.ones(self.state.Nvars)
        keys = self.state.get_rawkeys()
        for i,k in enumerate(keys):
            if k in scales:
                self.scales[i] = scales[k]
    '''
    Constitutes 1 MH step. Generates a proposal/transitions/catch errors & collisions.
    '''
    def step(self):
        while True:
            try:
                logp = self.state.get_logp(self.obs)
                proposal = self.generate_proposal() 
                if (proposal.priorHard()):
                    return False
                logp_proposal = proposal.get_logp(self.obs)
                if np.exp(logp_proposal-logp)>np.random.uniform():
                    self.state = proposal
                    return True
                return False
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                self.state.collisionGhostParams = proposal.get_params()
                return False

'''
smala MCMC coupled with rebound.
'''
class Smala(Mcmc):
    def __init__(self, initial_state, obs, eps, alp, ana=False):
        super(Smala,self).__init__(initial_state, obs)
        self.epsilon = eps
        self.alpha = alp
        self.analytical = ana

    '''
    Soft absolute metric, makes sure the eigenvalues are positive.
    '''
    def softabs(self, hessians):
        lam, Q = np.linalg.eig(-hessians)
        lam_twig = lam*1./np.tanh(self.alpha*lam)
        self.state.logp_dd_lam = lam_twig
        H_twig = np.dot(Q,np.dot(np.diag(lam_twig),Q.T))    
        return H_twig

    '''
    Generates a proposal based on the last state's logp, logp_d, logp_dd values.
    '''
    def generate_proposal(self, wt = None, eps = None):
        if(wt is None):
            wt = np.random.normal(0.,1.,self.state.Nvars)
        if(eps is None):
            eps = self.epsilon
        logp, logp_d, logp_dd = self.state.get_logp_d_dd(self.obs) 
        Ginv = np.linalg.inv(self.softabs(logp_dd))
        Ginvsqrt = np.linalg.cholesky(Ginv)   
        if((self.state.Ginv is None) or(self.state.Ginvsqrt is None)):
            self.state.Ginv = Ginv
            self.state.Ginvsqrt = Ginvsqrt
        mu = self.state.get_params() + (eps)**2 * np.dot(Ginv, logp_d)/2.
        newparams = mu + eps * np.dot(Ginvsqrt, wt)
        prop = self.state.deepcopy()
        prop.set_params(newparams)
        return prop

    '''
    Generates a proposal based on the last state's logp, logp_d, logp_dd values.
    '''
    def generate_proposal_analytical(self, eps= None):
        if(eps is None):
            eps = self.epsilon
        logp, logp_d, logp_dd = self.state.get_logp_d_dd_analytical(self.obs) 
        Ginv = np.linalg.inv(self.softabs(logp_dd))
        Ginvsqrt = np.linalg.cholesky(Ginv)   
        if((self.state.Ginv is None) or (self.state.Ginvsqrt is None)):
            self.state.Ginv = Ginv
            self.state.Ginvsqrt = Ginvsqrt
        mu = self.state.get_params() + (eps)**2 * np.dot(Ginv, logp_d)/2.
        newparams = mu + eps * np.dot(Ginvsqrt, np.random.normal(0.,1.,self.state.Nvars))
        prop = self.state.deepcopy()
        prop.set_params(newparams)
        return prop

    '''
    Calculates the transition probability given from a state to another.
    '''
    def transitionProbability(self,state_from, state_to, eps = None):
        if(eps is None):
            eps = self.epsilon
        logp, logp_d, logp_dd = state_from.get_logp_d_dd(self.obs)
        if(state_from.Ginv is None):
            state_from.Ginv = np.linalg.inv(self.softabs(logp_dd))
        mu = state_from.get_params() + (eps)**2 * np.dot(state_from.Ginv, logp_d)/2.
        return stats.multivariate_normal.logpdf(state_to.get_params(),mean=mu, cov=(eps)**2*state_from.Ginv)
        
    '''
    Constitutes 1 smala step. Generates a proposal/transitions/catch errors & collisions.
    '''
    def step(self):
        while True:
            try:
                stateStar = self.generate_proposal()
                if (stateStar.priorHard()):
                    return False
                q_ts_t = self.transitionProbability(self.state, stateStar)
                q_t_ts = self.transitionProbability(stateStar, self.state)
                break
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                self.state.collisionGhostParams = stateStar.get_params()
                return False
            except np.linalg.linalg.LinAlgError as err:
                print "np.linalg.linalg.LinAlgErrorhas occured, investigate later..."
                print stateStar.get_params()
                print self.state.get_params()
                quit()
        if np.exp(stateStar.logp-self.state.logp+q_t_ts-q_ts_t) > np.random.uniform():
            self.state = stateStar
            return True
        return False

'''
adaptive smala MCMC coupled with rebound.
'''
class Adaptsmala(Smala):
    def __init__(self, initial_state, obs, eps, alp, beta=10., rho=0.5, gamma=1.5):
        super(Adaptsmala,self).__init__(initial_state, obs, eps, alp)
        self.b = beta
        self.rh = rho
        self.gam = gamma

    def compute_new_eps(self, wt, state):
        #Assuming startin eps is reasonable eps
        eps = self.epsilon
        #Maximum 20 iterations, if it goes longer, give up(?)
        for i in range(20):
            delta_eps = np.absolute(self.compute_err(eps, wt, state))
            if(delta_eps > self.b):
                eps = self.rh*eps
            else:
                if(delta_eps < self.gam):
                    return eps
                else:
                    eps = 0.95 * (self.gam / delta_eps)**(1./3.) * eps

    def compute_err(self, eps, wt, state):
        logp, logp_d = state.get_logp_d(self.obs)
        if((state.logp_dd is None)):
            logp, logp_d, logp_d_dd = state.get_logp_d_dd(self.obs)
        if((state.Ginvsqrt is None)):
            state.Ginv = np.linalg.inv(self.softabs(state.logp_dd))
            state.Ginvsqrt = np.linalg.cholesky(state.Ginv)
        trial_state = self.generate_proposal(wt, eps)
        logp_star, logp_d_star = trial_state.get_logp_d(self.obs)
        r = np.dot(state.Ginvsqrt, (logp_d+logp_d_star))
        delta_err = -logp + logp_star - (eps/2.)*np.dot(wt, r) - (eps*eps/8.)*np.dot(r,r)
        return delta_err
        
    '''
    Constitutes 1 smala step. Generates a proposal/transitions/catch errors & collisions.
    '''
    def step_adapt(self):
        while True:
            try:
                wt = np.random.normal(0.,1.,self.state.Nvars)
                forwardeps = self.compute_new_eps(wt, self.state)
                stateStar = self.generate_proposal()
                if (stateStar.priorHard()):
                    return False
                backwardseps = self.compute_new_eps(wt, stateStar)
                q_ts_t = self.transitionProbability(self.state, stateStar, forwardeps)
                q_t_ts = self.transitionProbability(stateStar, self.state, backwardseps)
                break
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                self.state.collisionGhostParams = stateStar.get_params()
                return False
            except np.linalg.linalg.LinAlgError as err:
                print "np.linalg.linalg.LinAlgErrorhas occured, investigate later..."
                print stateStar.get_params()
                print self.state.get_params()
                quit()
        if np.exp(stateStar.logp-self.state.logp+q_t_ts-q_ts_t) > np.random.uniform():
            self.state = stateStar
            return True
        return False


class Alsmala(Smala):
    def __init__(self, initial_state, obs, eps, alp):
        super(Alsmala,self).__init__(initial_state, obs, eps, alp)

    def generate_proposal_mala(self):
        logp, logp_d, logp_dd = self.state.get_logp(self.obs), self.state.logp_d, self.state.logp_dd
        Ginv = np.linalg.inv(self.softabs(logp_dd))
        Ginvsqrt = np.linalg.cholesky(Ginv)   

        mu = self.state.get_params() + (self.epsilon)**2 * np.dot(Ginv, logp_d)/2.
        newparams = mu + self.epsilon * np.dot(Ginvsqrt, np.random.normal(0.,1.,self.state.Nvars))
        prop = self.state.deepcopy()
        prop.set_params(newparams)
        prop.logp_d = logp_d
        prop.logp_dd = logp_dd
        return prop

    def transitionProbability_mala(self,state_from, state_to):
        logp, logp_d, logp_dd = state_from.get_logp(self.obs), state_from.logp_d, state_from.logp_dd 
        Ginv = np.linalg.inv(self.softabs(logp_dd))
        mu = state_from.get_params() + (self.epsilon)**2 * np.dot(Ginv, logp_d)/2.
        return stats.multivariate_normal.logpdf(state_to.get_params(),mean=mu, cov=(self.epsilon)**2*Ginv)

    def step_mala(self):
        while True:
            try: 
                stateStar = self.generate_proposal_mala()
                if (stateStar.priorHard()):
                    return False
                q_ts_t = self.transitionProbability_mala(self.state, stateStar)
                q_t_ts = self.transitionProbability_mala(stateStar, self.state)
                break
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                self.state.collisionGhostParams = stateStar.get_params()
                return False
            except np.linalg.linalg.LinAlgError as err:
                print "np.linalg.linalg.LinAlgErrorhas occured, investigate later..."
                print stateStar.get_params()
                print self.state.get_params()
                quit()
        if np.exp(stateStar.logp-self.state.logp+q_t_ts-q_ts_t) > np.random.uniform():
            self.state = stateStar
            return True
        return False

class Hmc(Mcmc):
    def __init__(self, initial_state, obs, delt, l, masses):
        super(Hmc,self).__init__(initial_state, obs)
        self.delta = delt
        self.L = l
        self.momentum_vec = None
        self.new_momentum_vec = None
        self.old_K = 0.0
        self.new_K = 0.0
        self.mass_vector = masses
        self.temperature_scale = None

    def leap_frog(self):
        prop = self.state.deepcopy()
        assert(len(self.mass_vector) == len(self.momentum_vec))
        minv = np.reciprocal(self.mass_vector)
        self.new_momentum_vec = self.momentum_vec - 0.5*self.delta*np.multiply(minv,-self.state.logp_d)
        #print self.new_momentum_vec
        #print "momentum updated to ^"
        for i in range(self.L):
            q = prop.get_params() + np.multiply(minv,self.new_momentum_vec)*self.delta
            #print q
            #print "position(q) updated to ^"
            prop.set_params(q)
            logp, logp_d = prop.get_logp_d(self.obs)
            if(i != self.L-1):
                self.new_momentum_vec = self.new_momentum_vec - self.delta*np.multiply(minv, -logp_d)
                prop.logp_d = None
                #prop.logp = None
        self.new_momentum_vec = self.new_momentum_vec - 0.5*self.delta*np.multiply(minv, -prop.logp_d)
        self.new_momentum_vec = -self.new_momentum_vec
        return prop

    def generate_proposal(self):
        self.state.get_logp_d(self.obs)
        if(self.temperature_scale is not None):
            self.momentum_vec = self.temperature_scale * np.random.normal(size=(self.state.Nvars))
        else:
            self.momentum_vec = np.random.normal(size=(self.state.Nvars))
        self.new_momentum_vec = 0.0
        proposal_state = self.leap_frog()
        return proposal_state

    def step(self):
        while True:
            try:
                new_state = self.generate_proposal()
                if (new_state.priorHard()):
                    return False
                break
            except rebound.Encounter as err:
                print "Collision! {t}".format(t=datetime.utcnow())
                #self.state.collisionGhostParams = new_state.get_params()
                return False
        #self.new_K = np.dot(self.new_momentum_vec, self.new_momentum_vec)
        #self.old_K = np.dot(self.momentum_vec, self.momentum_vec)
        massive_matrix = np.linalg.inv(np.diag(self.mass_vector))
        self.new_K = np.dot(self.new_momentum_vec, np.dot(massive_matrix, self.new_momentum_vec))
        self.old_K = np.dot(self.momentum_vec, np.dot(massive_matrix, self.momentum_vec))
        #print "acceptance stuff"
        #print self.new_momentum_vec
        #print self.momentum_vec
        #print -self.new_K*0.5
        #print self.old_K*0.5
        #print np.exp(-new_state.logp + self.state.logp - self.new_K*0.5 + self.old_K*0.5)
        #print "aya"
        if (np.exp(new_state.logp - self.state.logp + self.new_K*0.5 - self.old_K*0.5) > np.random.uniform()):
            self.state = new_state
            return True
        return False


