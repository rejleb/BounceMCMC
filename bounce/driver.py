import rebound
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import observations
import state
import mcmc
import numpy as np
import hashlib
from datetime import datetime
from scipy import stats
import cPickle as pickle
import copy

################################################################################################
#This is a module which facilitates the use of the mcmc classes. 
#As such, it is structured differently.
#The idea is to create an MCMC bundle object which is driven by the driver class.
#These bundle objects simplify use and make it easy to save work via pickling.
################################################################################################

class McmcBundle(object):
    def __init__(self, mcmc, chain, chainlogp, clocktimes, collchain, obs, Niter, initial_state, label, dictionary, trimmedchain=None, trimmedchainlogp=None, actimes=None, eigenlist=None, is_emcee = False, Nwalkers = 32):
        self.mcmc = mcmc
        self.mcmc_is_emcee = is_emcee
        self.mcmc_Nwalkers = Nwalkers
        self.mcmc_eigenlist = eigenlist
        self.mcmc_chain = chain
        self.mcmc_chainlogp = chainlogp
        self.mcmc_clocktimes = clocktimes
        self.mcmc_obs = obs
        self.mcmc_Niter = Niter
        self.mcmc_initial_state = initial_state
        self.mcmc_label = label
        self.mcmc_trimmedchain = trimmedchain
        self.mcmc_trimmedchainlogp = trimmedchainlogp
        self.mcmc_actimes = actimes
        self.mcmc_hyperparams = dictionary
        self.mcmc_collision_counter = collchain
        self.mcmc_tness = None
        self.mcmc_dt = None
        self.mcmc_ess = None


#Two utility functions meant for the driver class
def auto_correlation(x):
    x = np.asarray(x)
    y = x-x.mean()
    result = np.correlate(y, y, mode='full')
    result = result[len(result)//2:]
    result /= result[0]
    return result

#Usefull in certain troubleshooting situations. Kept for record keeping.
def writing_to_log(obj, name, logging):
    if(logging):
        with open("log{r}".format(r=nameame),"a") as a:
            for index,value in np.ndenumerate(obj):
                a.write("{v} ".format(v=value))
            a.write("\n")
    else:
        a=None


#functions to be called by notebook/user
def run_mh(label, Niter, true_state, obs, scal, step, printing_every=400):
    mh = mcmc.Mh(true_state,obs)
    mh.set_scales(scal)
    mh.step_size = step
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    hyperparams = {'scale':scal, 'step':step}
    save_aux_before_run(label, true_state, Niter, hyperparams, h)
    chain = np.zeros((0,mh.state.Nvars))
    chainlogp = np.zeros(0)
    collchain = np.zeros((0,mh.state.Nvars))
    tries = 0
    clocktimes = [[]]
    clocktimes[0].append(datetime.utcnow())
    for i in range(Niter):
        if(mh.step()):
            tries += 1
        chainlogp = np.append(chainlogp,mh.state.get_logp(obs))
        chain = np.append(chain,[mh.state.get_params()],axis=0)
        if(mh.state.collisionGhostParams is not None):
            collchain = np.append(collchain, [mh.state.collisionGhostParams], axis=0)
            mh.state.collisionGhostParams = None
        if(i % printing_every == 1):
            print ("Progress: {p:.5}%, {n} accepted steps have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes[0].append(datetime.utcnow())
    clocktimes[0].append(datetime.utcnow())
    print("Acceptance rate: %.3f%%"%((tries/float(Niter))*100))
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(mh, chain, chainlogp, clocktimes, collchain, obs, Niter, true_state, label, hyperparams)
    return bundle, h

def run_emcee(label, Niter, true_state, obs, Nwalkers, scal, a=2, printing_every=400, listmode=False):
    ens = mcmc.Ensemble(true_state,obs,scales=scal,nwalkers=Nwalkers, a=a,listmode=listmode)
    h = hashlib.md5()
    #temporarily taken out for compatibility with ttvfaststate.py which does not have self.planets
    #h.update(str(true_state.planets))
    h.update(label)
    hyperparams = {'scale':scal, 'walkers':Nwalkers, 'a':a}
    #temporarily taken out for compatibility with ttvfaststate.py which does not have self.planets
    #save_aux_before_run(label, true_state, Niter, hyperparams, h)
    listchain = np.zeros((Nwalkers,ens.state.Nvars,0))
    listchainlogp = np.zeros((Nwalkers,0))
    collchain = np.zeros((0,ens.state.Nvars))
    tries=0
    clocktimes = [[]]
    clocktimes[0].append(datetime.utcnow())
    for i in range(int(Niter/Nwalkers)):
        if(ens.step()):
            for q in range(len(ens.states)):
                if(np.any(ens.previous_states[q] != ens.states[q])):
                    tries += 1
        listchainlogp = np.append(listchainlogp, np.reshape(ens.lnprob, (Nwalkers, 1)), axis=1)
        listchain = np.append(listchain, np.reshape(ens.states, (Nwalkers,ens.state.Nvars,1)),axis=2)
        if(len(ens.state.collisionGhostParams)!=0):
            for i in range(len(ens.state.collisionGhostParams)):
                c = ens.state.collisionGhostParams[i].deepcopy()
                collchain = np.append(collchain, [c.get_params()], axis=0)
        if (i%printing_every==1): 
            print ("Progress: {p:.5}%, time: {t}".format(p=100.*(float(i)/(Niter/Nwalkers)),t=datetime.utcnow()))
            clocktimes[0].append(datetime.utcnow())
    clocktimes[0].append(datetime.utcnow())
    print("Error(s): {e}".format(e=ens.totalErrorCount))
    print("Acceptance rate: %.3f%%"%(tries/(float(Niter))*100))
    chain = np.zeros((ens.state.Nvars,0))
    chainlogp = np.zeros(0)
    for i in range(Nwalkers):
        chain = np.append(chain, listchain[i], axis=1)
        chainlogp = np.append(chainlogp, listchainlogp[i])
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(ens, np.transpose(chain), chainlogp, clocktimes, collchain, obs, Niter, true_state, label, hyperparams, is_emcee=True, Nwalkers=Nwalkers)
    return bundle, h

def run_smala(label, Niter, true_state, obs, eps, alpha, printing_every = 40):
    smala = mcmc.Smala(true_state,obs, eps, alpha)
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    hyperparams = {'eps':eps, 'alpha':alpha}
    save_aux_before_run(label, true_state, Niter, hyperparams, h)
    chain = np.zeros((0,smala.state.Nvars))
    chainlogp = np.zeros(0)
    collchain = np.zeros((0,smala.state.Nvars))
    tries = 0
    clocktimes = [[]]
    eigenlist = np.zeros(0)
    clocktimes[0].append(datetime.utcnow())
    for i in range(Niter):
        if(smala.step()):
            tries += 1
        chainlogp = np.append(chainlogp,smala.state.get_logp(obs))
        chain = np.append(chain,[smala.state.get_params()],axis=0)
        if(smala.state.collisionGhostParams is not None):
            collchain = np.append(collchain, [smala.state.collisionGhostParams], axis=0)
            smala.state.collisionGhostParams = None
        la, Q = np.linalg.eig(-smala.state.logp_dd)
        lam = la*1./np.tanh(alpha*la)
        eigenlist = np.append(eigenlist,lam)
        if(i % printing_every == 1):
            print ("Progress: {p:.5}%, {n} accepted steps have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes[0].append(datetime.utcnow())
    clocktimes[0].append(datetime.utcnow())
    print("Acceptance rate: %.2f%%"%((tries/float(Niter))*100))
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(smala, chain, chainlogp, clocktimes, collchain, obs, Niter, true_state, label, hyperparams, eigenlist=eigenlist)
    return bundle, h

def run_adaptsmala(label, Niter, true_state, obs, eps, alpha, printing_every = 40):
    smala = mcmc.Adaptsmala(true_state,obs, eps, alpha)
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    hyperparams = {'eps':eps, 'alpha':alpha}
    save_aux_before_run(label, true_state, Niter, hyperparams, h)
    chain = np.zeros((0,smala.state.Nvars))
    chainlogp = np.zeros(0)
    collchain = np.zeros((0,smala.state.Nvars))
    tries = 0
    clocktimes = [[]]
    eigenlist = np.zeros(0)
    clocktimes[0].append(datetime.utcnow())
    for i in range(Niter):
        if(smala.step()):
            tries += 1
        chainlogp = np.append(chainlogp,smala.state.get_logp(obs))
        chain = np.append(chain,[smala.state.get_params()],axis=0)
        if(smala.state.collisionGhostParams is not None):
            collchain = np.append(collchain, [smala.state.collisionGhostParams], axis=0)
            smala.state.collisionGhostParams = None
        la, Q = np.linalg.eig(-smala.state.logp_dd)
        lam = la*1./np.tanh(alpha*la)
        eigenlist = np.append(eigenlist,lam)
        if(i % printing_every == 1):
            print ("Progress: {p:.5}%, {n} accepted steps have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes[0].append(datetime.utcnow())
    clocktimes[0].append(datetime.utcnow())
    print("Acceptance rate: %.2f%%"%((tries/float(Niter))*100))
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(smala, chain, chainlogp, clocktimes, collchain, obs, Niter, true_state, label, hyperparams, eigenlist=eigenlist)
    return bundle, h

def pre_eps_smala(true_state, obs, eps, alpha, Niter):
    smala = mcmc.Smala(true_state,obs, eps, alpha)
    print "Trying out eps = {e}".format(e = eps)
    tries = 0
    for i in range(Niter):
        while smala.step()==False:
            tries += 1
        tries += 1
    print "Acc. Rate was {a}".format(a=(float(Niter)/tries))
    if((0.52<=(float(Niter)/tries)) and (0.68>=(float(Niter)/tries))):
        return eps
    elif(0.52>(float(Niter)/tries)):
        mod = 0
        while(mod<=0):
            mod = np.random.normal(loc=0.065, scale=0.025)*8.*np.abs((float(Niter)/tries)-0.6)
        return preEpsSMALA(true_state, obs, eps-mod, alpha, Niter)
    elif(0.68<(float(Niter)/tries)):
        mod = 0
        while(mod<=0):
            mod = np.random.normal(loc=0.065, scale=0.025)*8.*np.abs((float(Niter)/tries)-0.6)
        return preEpsSMALA(true_state, obs, eps+mod, alpha, Niter)

def run_alsmala(label, Niter, true_state, obs, eps, alpha, bern_a, bern_b, printing_every = 40):
    alsmala = mcmc.Alsmala(true_state,obs, eps, alpha)
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    hyperparams = {'eps':eps, 'alpha':alpha, 'bern_a':bern_a, 'bern_b':bern_b}
    save_aux_before_run(label, true_state, Niter, hyperparams, h)
    chain = np.zeros((0,alsmala.state.Nvars))
    chainlogp = np.zeros(0)
    collchain = np.zeros((0,alsmala.state.Nvars))
    eigenlist = np.zeros(0)
    tries = 0
    clocktimes = [[]]
    clocktimes[0].append(datetime.utcnow())
    for i in range(Niter):
        if( (1-bern_b)*(np.exp(-bern_a*(i)/Niter))+bern_b >np.random.uniform()):
            if(alsmala.step()):
                tries +=1
        else:
            if(alsmala.step_mala()):
                tries += 1
        chainlogp = np.append(chainlogp,alsmala.state.get_logp(obs))
        chain = np.append(chain,[alsmala.state.get_params()],axis=0)
        if(alsmala.state.collisionGhostParams is not None):
            collchain = np.append(collchain, [alsmala.state.collisionGhostParams], axis=0)
            alsmala.state.collisionGhostParams = None
        la, Q = np.linalg.eig(-alsmala.state.logp_dd)
        lam = la*1./np.tanh(alpha*la)
        eigenlist = np.append(eigenlist,lam)
        if(i % printing_every == 1):
            print ("Progress: {p:.5}%, {n} accepted steps have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes[0].append(datetime.utcnow())
    clocktimes[0].append(datetime.utcnow())
    print("Acceptance rate: %.2f%%"%((tries/float(Niter))*100))
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(alsmala, chain, chainlogp, clocktimes, collchain, obs, Niter, true_state, label, hyperparams, eigenlist=eigenlist)
    return bundle, h

def run_hmc(label, Niter, true_state, obs, delt, L, masses, temperatures=None, printing_every = 40):
    hmc = mcmc.Hmc(true_state, obs, delt, L, masses)
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    hyperparams = {'delt':delt, 'L':L, 'mass':masses}
    save_aux_before_run(label, true_state, Niter, hyperparams, h)
    chain = np.zeros((0,hmc.state.Nvars))
    chainlogp = np.zeros(0)
    collchain = np.zeros((0,hmc.state.Nvars))
    tries = 0
    clocktimes = [[]]
    clocktimes[0].append(datetime.utcnow())
    for i in range(Niter):
        if(temperatures is not None):
            print temperatures
            hmc.temperature_scale = temperatures[Niter]
        if(hmc.step()):
            tries += 1
        chainlogp = np.append(chainlogp,hmc.state.get_logp(obs))
        chain = np.append(chain,[hmc.state.get_params()],axis=0)
        if(hmc.state.collisionGhostParams is not None):
            collchain = np.append(collchain, [hmc.state.collisionGhostParams], axis=0)
            hmc.state.collisionGhostParams = None
        if(i % printing_every == 1):
            print ("Progress: {p:.5}%, {n} accepted steps have been made, time: {t}".format(p=100.*(float(i)/Niter),t=datetime.utcnow(),n=tries))
            clocktimes[0].append(datetime.utcnow())
    clocktimes[0].append(datetime.utcnow())
    print("Acceptance rate: %.2f%%"%((tries/float(Niter))*100))
    print "The id of the simulation is: {r}".format(r=h.hexdigest())
    print "The end time of the simulation is {r}".format(r=datetime.utcnow())
    bundle = McmcBundle(hmc, chain, chainlogp, clocktimes, collchain, obs, Niter, true_state, label, {'delt':delt, 'L':L, 'mass':masses})
    return bundle, h

#Idea: PCGSMALA
#To be implemented later
def run_PCGSMALA():
    pass

def create_obs(state, npoint, err, errVar, t):
    obs = observations.FakeObservation(state, Npoints=npoint, error=err, errorVar=errVar, tmax=(t))
    return obs

def read_obs(filen):
    obs = observations.Observation_FromFile(filename=filen, Npoints=100)
    return obs

def save_obs(obs, true_state, label):
    col1 = obs.t/1.720e-2
    col2 = obs.rv/3.355e-5
    col3 = obs.err/3.355e-5
    h = hashlib.md5()
    h.update(str(true_state.planets))
    h.update(label)
    np.savetxt('obs_{ha}.vels'.format(ha=h.hexdigest()), np.c_[col1, col2, col2])

def plot_obs(true_state, obs, size, name='Name_left_empty', save=False):
    fig = plt.figure(figsize=(size[0],size[1]))
    font = FontProperties()
    font.set_family('serif')
    font.set_style('italic')
    ax = plt.subplot(111)
    ax.plot(*true_state.get_rv_plotting(obs), color="blue")
    plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax.set_xticklabels([])
    ax.set_ylabel("$Initial RV$", fontsize=28)
    ax.tick_params(axis='both', labelsize=18)
    plt.grid()
    frame2=fig.add_axes([0.125, -0.17, 0.775, 0.22])
    plt.tick_params(axis='both', labelsize=18)  
    frame2.set_ylabel("$Res. RV$", fontsize=28)
    frame2.set_xlabel("$Time$", fontsize=28)      
    plt.errorbar(obs.t, true_state.get_rv(obs.t)-obs.rv, yerr=obs.err, fmt='.r')
    frame2.locator_params(nbins=3, axis='y')
    plt.grid()
    if(save):
        plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
        plt.close('all')

def plot_chains(bundle, size, name='Name_left_empty', save=False):
    fig = plt.figure(figsize=(size[0],size[1]))
    font = FontProperties()
    font.set_family('serif')
    font.set_style('italic')
    mcmc, chain, chainlogp = bundle.mcmc, bundle.mcmc_chain, bundle.mcmc_chainlogp
    for i in range(mcmc.state.Nvars):
        ax = plt.subplot(mcmc.state.Nvars+1,1,1+i)
        ax.set_ylabel(mcmc.state.get_keys()[i])
        ax.tick_params(axis='x', labelbottom='off')
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=18)
        ax.locator_params(axis='y', nbins=3)
        ax.plot(chain[:,i])
    ax = plt.subplot(mcmc.state.Nvars+1,1,mcmc.state.Nvars+1)
    ax.set_ylabel("$\log \, p$")
    ax.set_xlabel("$Iterations$")
    ax.yaxis.label.set_size(28)
    ax.xaxis.label.set_size(28)
    ax.tick_params(axis='both', labelsize=18)
    ax.locator_params(axis='y', nbins=3)
    ax.plot(chainlogp)
    if(save):
        plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
        plt.close('all')

def ttv_from_times(t1_times):
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.arange(len(t1_times)),t1_times)
    diff = (np.arange(len(t1_times))*slope + intercept) - np.asarray(t1_times)
    return t1_times, diff

def plot_ttv_results(bundle, Ntrails, size):
    return_trimmed_results(bundle, 50, [1,1], 0.5, plotting=False)
    print "Begin plotting..."
    Niter, mcmc, chain, chainlogp, true_state, obs, is_emcee, Nwalkers = bundle.mcmc_Niter, bundle.mcmc, bundle.mcmc_chain, bundle.mcmc_chainlogp, bundle.mcmc_initial_state, bundle.mcmc_obs, bundle.mcmc_is_emcee, bundle.mcmc_Nwalkers
    trimmedchain = bundle.mcmc_trimmedchain
    list_of_states = []
    for i in range(len(trimmedchain)):
        s = mcmc.state.deepcopy()
        s.set_params(chain[i])
        list_of_states.append(s)

    fig = plt.figure(figsize=(size[0],size[1]))
    #font = FontProperties()
    #font.set_family('serif')
    #font.set_style('italic')
    ax = plt.subplot(111)
    selected = np.sort(np.random.choice(len(list_of_states), Ntrails))
    print "Selected some {nt} samples to plot.".format(nt=Ntrails)
    for j in range(len(selected)):
        a = list_of_states[selected[j]]
        ax.plot(*a.get_ttv_plotting(obs), alpha=0.28, color="darkolivegreen")
        
    averageRandomState = mcmc.state.deepcopy()
    averageRandomChain = np.average(chain, axis=0)
    averageRandomState.set_params(averageRandomChain)
    ax.plot(*true_state.get_ttv_plotting(obs), color="purple")
    #plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax.set_xticklabels([])
    plt.grid()
    ax2=fig.add_axes([0.125, -0.63, 0.775, 0.7]) 
    #ax.set_ylabel("Initial TTV", fontproperties=font, fontsize=26)
    #ax2.set_ylabel("Average Result TTV", fontproperties=font, fontsize=26)
    ax.yaxis.label.set_size(26)
    ax2.yaxis.label.set_size(26)
    ax.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    plt.plot(*averageRandomState.get_ttv_plotting(obs), alpha=0.99,color="black")
    print "Resulting average params state (randomly sampledriver.ind):"
    #print averageRandomState.get_keys()
    print averageRandomState.get_params()
    #plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax2.set_xticklabels([])
    plt.grid()
    plt.close('all')
    return fig


def return_trimmed_results(bundle, Ntrails, size, burn_in_fraction, take_every_n=1, name='Name_left_empty', save=False, plotting=True):
    Niter, mcmc, chain, chainlogp, true_state, obs, is_emcee, Nwalkers = bundle.mcmc_Niter, bundle.mcmc, bundle.mcmc_chain, bundle.mcmc_chainlogp, bundle.mcmc_initial_state, bundle.mcmc_obs, bundle.mcmc_is_emcee, bundle.mcmc_Nwalkers
    if(bundle.mcmc_is_emcee):
        assert (Niter/Nwalkers*burn_in_fraction % 1==0.0),"Burn in fraction must divide Niter/Nwalkers!"
    averageRandomChain = np.zeros(mcmc.state.Nvars)
    Allocationsize = int(float(1./take_every_n)*Niter*(1.-burn_in_fraction))
    list_of_states = []
    list_of_chainlogp = np.zeros(Allocationsize)
    iteration = 0
    if(is_emcee):
        for i in range(Nwalkers):
            for c in range(int( ((i)*Niter/Nwalkers)+Niter/Nwalkers*burn_in_fraction), (i+1)*Niter/Nwalkers):
                if(c%take_every_n==0):
                    s = mcmc.state.deepcopy()
                    s.set_params(chain[c])
                    list_of_states.append(s)
                    #list_of_chainlogp = np.append(list_of_chainlogp, chainlogp[c])
                    list_of_chainlogp[iteration] = chainlogp[c]
                    iteration += 1
                    averageRandomChain += chain[c]
    else:
        for c in range(int(Niter*burn_in_fraction), Niter):
            if(c%take_every_n==0):
                s = mcmc.state.deepcopy()
                s.set_params(chain[c])
                list_of_states.append(s)
                #list_of_chainlogp = np.append(list_of_chainlogp, chainlogp[c])
                list_of_chainlogp[iteration] = chainlogp[c]
                iteration += 1
                averageRandomChain += chain[c]
    print "Eliminated burn in, sampled every {n}.".format(n=take_every_n)
    list_of_results = np.zeros((Allocationsize,mcmc.state.Nvars))    
    for i in range(len(list_of_states)):
        if(i%1000==999999):
            print "Appending results: {i}%".format(i=float(i)/len(list_of_states))
        #list_of_results = np.append(list_of_results, [list_of_states[i].get_params()], axis=0)
        list_of_results[i] = list_of_states[i].get_params()
    bundle.mcmc_trimmedchain, bundle.mcmc_trimmedchainlogp  = list_of_results, list_of_chainlogp
    if(plotting):
        plot_trimmed_results(bundle, Ntrails, size, name, save)

     
def plot_trimmed_results(bundle, Ntrails, size, name='', save=False):
    Niter, mcmc, chain, chainlogp, true_state, obs, is_emcee, Nwalkers = bundle.mcmc_Niter, bundle.mcmc, bundle.mcmc_chain, bundle.mcmc_chainlogp, bundle.mcmc_initial_state, bundle.mcmc_obs, bundle.mcmc_is_emcee, bundle.mcmc_Nwalkers
    trimmedchain = bundle.mcmc_trimmedchain
    list_of_states = []
    for i in range(len(trimmedchain)):
        s = mcmc.state.deepcopy()
        s.set_params(chain[i])
        list_of_states.append(s)

    fig = plt.figure(figsize=(size[0],size[1]))
    font = FontProperties()
    font.set_family('serif')
    font.set_style('italic')
    ax = plt.subplot(111)
    selected = np.sort(np.random.choice(len(list_of_states), Ntrails))
    for j in range(len(selected)):
        a = list_of_states[selected[j]]
        ax.plot(*a.get_rv_plotting(obs), alpha=0.28, color="darkolivegreen")

    print "Selected some {nt} samples to plot.".format(nt=Ntrails)

    averageRandomState = mcmc.state.deepcopy()
    averageRandomChain = np.average(chain, axis=0)
    averageRandomState.set_params(averageRandomChain)
    ax.plot(*true_state.get_rv_plotting(obs), color="purple")
    plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax.set_xticklabels([])
    plt.grid()
    ax2=fig.add_axes([0.125, -0.63, 0.775, 0.7]) 
    ax.set_ylabel("Initial RV", fontproperties=font, fontsize=26)
    ax2.set_ylabel("Average Result RV", fontproperties=font, fontsize=26)
    ax.yaxis.label.set_size(26)
    ax2.yaxis.label.set_size(26)
    ax.tick_params(axis='both', labelsize=18)
    ax2.tick_params(axis='both', labelsize=18)
    plt.plot(*averageRandomState.get_rv_plotting(obs), alpha=0.99,color="black")
    print "Resulting average params state (randomly sampledriver.ind):"
    print averageRandomState.get_keys()
    print averageRandomState.get_params()
    plt.errorbar(obs.t, obs.rv, yerr=obs.err, fmt='.r')
    ax2.set_xticklabels([])
    plt.grid()
    ax3=fig.add_axes([0.125, -0.9, 0.775, 0.23])  
    ax3.tick_params(axis='both', labelsize=18)  
    ax3.yaxis.label.set_size(26)
    ax3.set_ylabel("Res. RV", fontproperties=font, fontsize=26) 
    ax3.locator_params(nbins=3, axis='y')
    ax3.xaxis.label.set_size(26)
    ax3.set_xlabel("Time", fontproperties=font, fontsize=26)    
    plt.errorbar(obs.t, averageRandomState.get_rv(obs.t)-obs.rv, yerr=obs.err, fmt='.r')
    plt.grid()
    if(save):
        plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
    plt.close('all')
    return fig

#disable this function for now since installing corners on scinet is annoying, this is the fastest way to not do this.
def plot_corners(bundle, name='Name_left_empty', save=False):
    import corner
    chain, mcmc, true_state = bundle.mcmc_chain, bundle.mcmc, bundle.mcmc_initial_state
    somestate = mcmc.state.deepcopy()
    figure = corner.corner(chain, labels=somestate.get_keys(), plot_contours=False, truths=true_state.get_params(),label_kwargs={"fontsize":33},max_n_ticks=4)
    if(save):    
        plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')

def plot_ACTimes(bundle, size, fraction=1.0,name='Name_left_empty', save=False, plotting=False): 
    chain, mcmc, niter = bundle.mcmc_trimmedchain, bundle.mcmc, bundle.mcmc_Niter
    somestate = mcmc.state.deepcopy()
    actimes = np.zeros((somestate.Nvars, 2))
    if(plotting):
        fig = plt.figure(figsize=(size[0],size[1]))
        font = FontProperties()
        font.set_family('serif')
        font.set_style('italic')
    #fig.suptitle('Autocorelation', fontsize=12)
    for i in range(somestate.Nvars):
        if(plotting):
            ax = plt.subplot(somestate.Nvars+1,1,1+i)
            ax.set_ylabel(somestate.get_keys()[i])
            ax.yaxis.label.set_size(28)
            ax.tick_params(axis='x', labelbottom='off')
            ax.tick_params(axis='both', labelsize=13)
            ax.locator_params(axis='y', nbins=3)
            if(bundle.mcmc_is_emcee):
                plt.xlim([0, niter/bundle.mcmc_Nwalkers*fraction])
            else:
                plt.xlim([0, niter*fraction])
            plt.grid()
        if(bundle.mcmc_is_emcee):
            chain_from_emcee = bundle.mcmc_chain
            Nwalkers = bundle.mcmc_Nwalkers
            Niter = bundle.mcmc_Niter
            temp = np.zeros(Niter/Nwalkers)
            x = 0
            for k in range(Nwalkers):
                for p in range(Niter/Nwalkers):
                    temp[p] = chain_from_emcee[(Niter/Nwalkers)*k+p,i]
                r = auto_correlation(temp)
                s = np.sum(r)
                if(plotting):
                    ax.plot(r, alpha=0.39, color="darkolivegreen")
                for j in range(len(r)):
                    if(r[j] <0.5):
                        actimes[i,0] += j
                        break
                for j in range(len(r)):
                    if(r[j] >= 0.0):
                        actimes[i,1] += r[j]
                    else:
                        break
            actimes[i,0] /= Nwalkers
            actimes[i,1] /= Nwalkers
        else:
            r = auto_correlation(chain[:,i])
            if(plotting):
                ax.plot(r)
            for j in range(len(r)):
                if(r[j] < 0.5):
                    actimes[i,0] = j
                    break
            for j in range(len(r)):
                if(r[j] >= 0.0):
                    actimes[i,1] += r[j]
                else:
                    break
        print "AC time {t}".format(t=actimes[i])
    if(plotting):    
        ax.set_xlabel("$k-lag$")
        ax.xaxis.label.set_size(28)
        ax.tick_params(axis='x', labelbottom='on')
        plt.xlim([0, len(r)*fraction])
    if(save):
        plt.savefig('mcmcplots/{n}.png'.format(n=name), bbox_inches='tight')
        plt.close('all')
    bundle.mcmc_actimes = actimes

#Old code kept around until I am sure I don't need it (older/oldest versions found in github).
def inLinePlotEmceeAcTimes(bundle, size):
    chain, Niter, Nwalkers, mcmc = bundle.mcmc_chain, bundle.mcmc_Niter, bundle.mcmc_Nwalkers, bundle.mcmc
    somestate = mcmc.state.deepcopy()
    actimes = np.zeros(somestate.Nvars)
    fig = plt.figure(figsize=(size[0],size[1]))
    #fig.suptitle('Autocorelation', fontsize=12)
    for i in range(somestate.Nvars):
        ax = plt.subplot(somestate.Nvars+1,1,1+i)
        ax.set_ylabel(somestate.get_keys()[i])
        ax.yaxis.label.set_size(28)
        ax.tick_params(axis='both', labelsize=13)
        ax.locator_params(axis='y', nbins=3)
        temp = np.zeros(Niter/Nwalkers)
        x = 0
        for k in range(Nwalkers):
            for p in range(Niter/Nwalkers):
                temp[p] = chain[(Niter/Nwalkers)*k+p,i]
            y = auto_correlation(temp)
            ax.plot(y, alpha=0.18, color="darkolivegreen")
            for j in range(len(y)):
                if(y[j] <0.5):
                    actimes[i] += j
                    break
        actimes[i] /= Nwalkers
        print "AC time {t}".format(t=actimes[i])
    return actimes

def efficacy(Niter, AC, clockTimes):
    A = np.transpose(AC)[0]
    dt = 0
    for i in range(len(clockTimes)):
        dt += (clockTimes[i][len(clockTimes[i])-1]-clockTimes[i][1]).total_seconds()
    return (Niter/(dt*np.amax(A))), A, dt

def tness(bundle):
    Niter, AC, clockTimes = bundle.mcmc_Niter, bundle.mcmc_actimes, bundle.mcmc_clocktimes
    B = np.transpose(AC)[1]
    ess = Niter/(2.*B-1.)
    dt = 0
    for i in range(len(clockTimes)):
        dt += (clockTimes[i][len(clockTimes[i])-1]-clockTimes[i][0]).total_seconds()
    bundle.mcmc_tness = np.min(ess)/dt
    bundle.mcmc_ess = ess
    bundle.mcmc_dt = dt 


def compare_cdf(list_o_bundles, size, fontsize=28):
    mcmc = list_o_bundles[0].mcmc
    for i in range(len(np.transpose(list_o_bundles[0].mcmc_trimmedchain))):
        fig = plt.figure(figsize=(size[0],size[1]))
        ax = plt.subplot(111)
        for b in range(len(list_o_bundles)):
            chain = list_o_bundles[b].mcmc_trimmedchain
            legend_label = list_o_bundles[b].mcmc_label
            plt.plot(sorted(np.transpose(chain)[i]), np.linspace(0,1, len(np.transpose(chain)[i])), label=legend_label)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        ax.xaxis.label.set_size(fontsize)
        ax.yaxis.label.set_size(fontsize)
        ax.set_xlabel(mcmc.state.get_keys()[i])
        plt.ylabel('Fractional CDF')
        
def calc_kstatistic(chain1, chain2):
    for i in range(len(np.transpose(chain1))):
        print stats.ks_2samp(np.transpose(chain1)[i], np.transpose(chain2)[i])

def fuse_bundle(bun1, bun2, mode=0):
    assert(bun1.mcmc_is_emcee == bun2.mcmc_is_emcee)
    bun1.mcmc_Nwalkers += bun2.mcmc_Nwalkers
    #bun1.mcmc_eigenlist = np.append(bun1.mcmc_eigenlist, bun2.mcmc_eigenlist)
    #This line below is the general use version for later. Data used old driver, so we use two lines below instead for now.
    #bun1.mcmc_clocktimes = np.append(bun1.mcmc_clocktimes, bun2.mcmc_clocktimes, axis=0)
    #bun1.mcmc_clocktimes = np.append(bun1.mcmc_clocktimes, bun2.mcmc_clocktimes, axis=0)
    #assert(bun1.mcmc_obs == bun2.mcmc_obs)
    #initial condition ignores/not checked for now. May be implemented in future update.
    #Same for label
    #Assume fusion must happens before trimming 
    bun1.mcmc_hyperparams = np.append(bun1.mcmc_hyperparams, bun2.mcmc_hyperparams)
    if(mode==0):
        return_trimmed_results(bun2, 50, [1,1], 0.5, take_every_n=1, plotting=False)
        bun1.mcmc_trimmedchain = np.append(bun1.mcmc_trimmedchain, bun2.mcmc_trimmedchain, axis=0)
        bun1.mcmc_trimmedchainlogp = np.append(bun1.mcmc_trimmedchainlogp, bun2.mcmc_trimmedchainlogp)
        bun1.mcmc_clocktimes.append(bun2.mcmc_clocktimes[0])
    else:
        return_trimmed_results(bun1, 50, [1,1], 0.5, take_every_n=1, plotting=False)
        return_trimmed_results(bun2, 50, [1,1], 0.5, take_every_n=1, plotting=False)
        bun1.mcmc_trimmedchain = np.append(bun1.mcmc_trimmedchain, bun2.mcmc_trimmedchain, axis=0)
        bun1.mcmc_trimmedchainlogp = np.append(bun1.mcmc_trimmedchainlogp, bun2.mcmc_trimmedchainlogp)
        t = bun1.mcmc_clocktimes
        t.append(bun2.mcmc_clocktimes[0])
        bun1.mcmc_clocktimes = t
    bun1.mcmc_Niter += bun2.mcmc_Niter
    bun1.mcmc_chain = np.append(bun1.mcmc_chain, bun2.mcmc_chain, axis=0)
    bun1.mcmc_chainlogp = np.append(bun1.mcmc_chainlogp, bun2.mcmc_chainlogp)
    return bun1


def load_data(name, h):
    return pickle.load( open( '{n}_{h}.bund'.format(n=name,h=h.hexdigest()) , "rb" ) )

def save_data(datbundle, h):
    name=datbundle.mcmc_label
    pickle.dump( datbundle, open( '{n}_{h}.bund'.format(n=name,h=h.hexdigest()) , "wb" ) )

def save_aux_before_run(name, initial_state, Niter, hyperparams, h):
    with open('{n}_{h}.aux'.format(h=h.hexdigest(), n=name), "w") as text_file:
        text_file.write('Label = '+ name +'\n')
        text_file.write('Initial state = '+ str(initial_state.planets) +'\n')
        text_file.write('Niter = '+ str(Niter) +'\n')
        text_file.write('Hyperparams = '+ str(hyperparams) +'\n')

def save_aux(bundle, h):
    name, initial_state, Niter, hyperparams = bundle.mcmc_label, bundle.mcmc_initial_state, bundle.mcmc_Niter, bundle.mcmc_hyperparams
    with open('{n}_{h}.aux'.format(h=h.hexdigest(), n=name), "w") as text_file:
        text_file.write('Label = '+ name +'\n')
        text_file.write('Initial state = '+ str(initial_state.planets) +'\n')
        text_file.write('Niter = '+ str(Niter) +'\n')
        text_file.write('Hyperparams = '+ str(hyperparams) +'\n')


def bundle_stats(bundle):
    #try:
        tness(bundle)
        print "Time normalized ESS: {tness}".format(tness=bundle.mcmc_tness)
        print "ESS array: {tness}".format(tness=bundle.mcmc_ess)
        print "CPU time in seconds: {tness}".format(tness=bundle.mcmc_dt)
        copy = np.copy(np.transpose(bundle.mcmc_trimmedchain))
        avg = np.zeros(bundle.mcmc.state.Nvars)
        std = np.zeros(bundle.mcmc.state.Nvars)
        ef = np.zeros(bundle.mcmc.state.Nvars)
        for i in range(bundle.mcmc.state.Nvars):
            avg[i] = np.mean(copy[i])
            std[i] = np.std(copy[i])
            ef[i] = (std[i]/(0.01*avg[i]))**2.
        print "For 1% Monte Carlo error on the mean we need ({st}/(0.01*{av}))**2. = {ef} effective samples".format(ef=ef, st=std[i], av=avg[i])
        #print "CPU time needed for convergence is {t}".format(t = ef/timenorm)
        for i in range(len(copy)):
             print "The credible interval/HDI is from {inner} to {out}".format(inner=sorted(copy[i])[int(len(copy[i])*0.025)], out=sorted(copy[i])[int(len(copy[i])*0.975)])
             print "The resulting parameter is {avrg}, + {upper}, - {lower}".format(avrg = avg[i], lower=avg[i]-sorted(copy[i])[int(len(copy[i])*0.025)], upper=sorted(copy[i])[int(len(copy[i])*0.975)]-avg[i])

    #except:
    #    print "Some quantities are missing! Cannot compute statistics."

def gelmanrubin_statistic(segments, bundle):
    chain, mcmc = bundle.mcmc_trimmedchain, bundle.mcmc
    chaint = np.transpose(chain)
    gelmanrubinresults = np.zeros(len(chaint))
    for i in range(len(chaint)):
        theta = np.mean(chaint[i])
        list_of_segments = np.split(chaint[i], segments)
        sum_b = 0
        sum_w = 0
        list_of_sigm_M = np.zeros(segments)
        list_of_theta_M = np.zeros(segments)
        for j in range(segments):
            theta_M = np.mean(list_of_segments[j])
            sigm_M = np.var(list_of_segments[j])
            list_of_theta_M[j] = theta_M
            list_of_sigm_M[j] = sigm_M
            sum_b += (theta_M-theta)**2.
            sum_w += sigm_M
        N = len(list_of_segments[0])
        B = (N/(segments-1.)) * sum_b
        W = sigm_M/segments
        V_hat = ((N-1.)/N)*W + ((segments+1.)/(segments*N))*B
        term1 = ((N-1.)/N)**2. * (1./segments) * np.var(list_of_sigm_M) 
        term2 = ((segments+1.)/(segments*N)) * (2./(segments-1.)) * B*B
        term3 = ((2*(segments+1.)*(N-1.))/(segments*N*N)) * (N/segments) * ( np.cov(list_of_sigm_M,np.square(list_of_theta_M))[0][1] - 2.*theta*np.cov(list_of_sigm_M,list_of_theta_M)[0][1] )
        var_V_hat = term1+term2+term3

        d_hat = 2*V_hat*V_hat/var_V_hat
        R_c = np.sqrt(V_hat*(d_hat+3)/(W*(d_hat+1)))
        gelmanrubinresults[i] = R_c
    return gelmanrubinresults
