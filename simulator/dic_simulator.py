import snap
from collections import namedtuple
import numpy as np
# np.random.seed(0)
"""
Simulate rounds of diffusion process under discrete IC model
to generate cascades from given seed sets.
Input
-----
snapGraph: PNEANet (snap package)
    nodes, edges, and optionally edge influence (activation probability) as edge attr.
obs_steps: int
    maximum diffusion time steps up to which the process is to be simulated.
"""

def generate_cascades(seed_sets, network, num_nodes, act_prob_constant, obs_steps=None):
    """
    Generate (Simulate) one cascade per seed set in seed_sets upto obs_steps.
    Parameters
    ----------
    seed_sets : list(list(int))
        list of seed sets
    Returns
    ----------
    list(np.array((None, 2)))
        list of cascades where each cascade is time ordered array of (active_nodeid, timestep)
    """
    cascades = list()
    for seeds in seed_sets:
        cascade = run_diffusion(seeds, network, num_nodes,
            act_prob_constant, rounds=obs_steps)
        cascades.append(cascade)
    return cascades


def simulate(seeds, num_simulations, network, num_nodes, act_prob_constant, obs_steps=None):
    """
    Simulate num_simulations cascades of given seed set upto obs_steps.
    Parameters
    ----------
    seeds : list(int)
        given seed set
    num_simulations : int
        number of times to simulate the IC model to obtain cascades starting from given seeds
    Returns
    ----------
    list(np.array((None, 2)))
        list of cascades where each cascade is time ordered array of (active_nodeid, timestep)
    """
    # replicate seedset for times = num_simulations 
    # seed_sets = np.repeat([seedset], repeats=num_simulations, axis=0)
    # cascades = generate_cascades(seed_sets, network, num_nodes, act_prob_constant, obs_steps)
    cascades = []
    for i in range(num_simulations):
        cascades.append(run_diffusion(seeds, network, num_nodes,
            act_prob_constant, rounds=obs_steps))
    return cascades


def run_diffusion(seeds, network, num_nodes, act_prob_constant, rounds=None):
    """
    Simulate diffusion process under IC model to generate
    one cascade from given seed set (rounds = maximimum diffusion steps).
    Parameters
    ----------
    seeds : list(int)
        given seed set
    rounds : int
        maimum number of time steps upto which the diffusion process is simulated.
        default = obs_steps set in init functions.
    Returns
    ----------
    np.array((None, 2))
        returns a cascade that is a time ordered array of (active_nodeid, timestep)
    """
    # seeds activated at step=1 (only for this function) but returned activation time (seeds=0)
    
    if rounds == None or rounds > num_nodes: rounds = num_nodes # diffuse till no new activations
    G = network
    activated_at = snap.TIntV(num_nodes)

    current_round = 1
    t_active = snap.TIntV()
    [t_active.Add(seed) for seed in seeds]
    # for i in range(num_nodes): activated_at[i] == -1
    for s in seeds:
        activated_at[int(s)] = current_round
    # cascade = snap.TIntIntH() # HashTable (node, discrete_activation_time)
    # for s in t_active:
    # cascade[s]=current_round # update hashmap

    while t_active and current_round <= rounds:
        tplus1_active = snap.TIntV()
        for s in t_active:
            NI = G.GetNI(s)
            for v in NI.GetOutEdges():
                EI = G.GetEI(s, v)
                p_sv = G.GetFltAttrDatE(EI, act_prob_constant)
                # if v not in cascade and np.random.rand() <= p_sv: # inactive and attempt is successful
                if activated_at[v]==0 and np.random.rand() <= p_sv: # inactive and attempt is successful
                    tplus1_active.Add(v)
                    activated_at[v] = current_round + 1
                    # cascade[v] = current_round + 1
        t_active = tplus1_active
        current_round += 1
     
    cascade = np.array(activated_at)
    active_u = np.nonzero(cascade)[0]
    active_t = cascade[active_u] - 1
    c = np.hstack([active_u.reshape(-1, 1), active_t.reshape(-1, 1)])
    del activated_at, active_u, active_t, cascade, t_active, tplus1_active
    return c[c[:,1].argsort()] # shape=(None, 2) for active node ids and discrete times in cascade (sorted by time)


