# -*- coding: utf-8 -*-
import numpy as np
# np.random.seed(0)
import sys, time
sys.path.append("../")
import snap
import pandas as pd
import pickle
from parameter_estimation import data_io
from simulator.greedy_clef import InfluenceEstimator
from simulator.sample_distribution import *


if __name__ == '__main__':
    start = time.time()
    """Input setting
    --------------------
    """
    out = 'all_tma/tma_E'
   
    pi_file = '../output/{}/pi.txt'.format(out)
    edges_file = '../output/{}/learned_graph.tsv'.format(out)
    idx2u_file = '../output/{}/idx2u.txt'.format(out)
    infl_users_file = '../output/{}/selected_influential_users.tsv'.format(out)
    
    log_file = '../output/{}/edge_intervention_log.txt'.format(out)
     
    fake_component = 0
    K_num_edges = [10, 20, 50, 80, 100]  # num of edges to remove 
    num_seed_sets = 100
    sample_seeds_from = 'users'  # 'users', 'infl'
    num_simulations = 10
    obs_steps = 50
    
    logf = open(log_file, 'w')
    sys.stdout = data_io.Runlog(sys.stdout, logf)  # This will go to stdout and the file out.txt
    
    """Print info
    """
    print('fake_component', fake_component)
    print('K_num_edges', K_num_edges)
    print('num_seedsets', num_seed_sets)
    print('sample_from', sample_seeds_from)
    print('num_sims', num_simulations)
    print('obs_steps', obs_steps)
    
    """Load graph and activation probabilities (learned and saved from parameter estimation)
    And Load selected influential users for seed set sampling if needed (estimated for fake component as specified in input) 
    --------------------
    """
    pi0, pi1, base_graph, idx2u, u2idx = data_io.load_estimated_parameters(pi_file, edges_file, idx2u_file)
    base_df = data_io.load_base_graph_as_dataframe(edges_file)
    # fake_influential_users = data_io.load_selected_infl_users(infl_users_file, component=fake_component)
    # true_influential_users = data_io.load_selected_infl_users(infl_users_file, component=1-fake_component)
    num_nodes = base_graph.GetNodes()
    assert num_nodes == len(idx2u), 'inconsistent num of nodes in base_graph and in idx2u'
    
    """Sample seed sets
    """
    if sample_seeds_from == 'users':
        seed_sets = sample_seed_sets(np.arange(len(idx2u)), alpha=2.5, num_seedsets=num_seed_sets)
    elif sample_seeds_from == 'fake_infl':
        seed_sets = sample_seed_sets(fake_influential_users, alpha=2.5, num_seedsets=num_seed_sets)
    elif sample_seeds_from == 'infl':
        seed_sets = sample_seed_sets(list(fake_influential_users) + list(true_influential_users), alpha=2.5, num_seed_sets=num_seed_sets)
    else:
        print('not implemented (sample_seeds_from).')
    
    """Run expected influence estimation on base_graph (without edge intervention)
    """
    influence_estimator = InfluenceEstimator(base_graph, num_nodes, 'act_prob_{}'.format(fake_component), obs_steps)
    exp_infl, std = influence_estimator.get_expected_influence_for_multiple_seed_sets(seed_sets, num_simulations)
    print('without intervention: {} \pm {}'.format(exp_infl, std))
    
    for K in K_num_edges:
        print("num of edges to remove in intervention = {}".format(K))
        """Run expected influence estimation (mic edge intervention)
        """
        puv_f = data_io.get_act_prob_from_dataframe(base_df, component=fake_component)  # df['act1']
        Z = [x for _,x in sorted(zip(puv_f, np.arange(len(base_df))), reverse=True)]
        discarded_edge_df = base_df.drop(Z[:K])
        mic_intervention_graph = data_io.convert_dataframe_to_base_graph(discarded_edge_df, idx2u)
        
        influence_estimator = InfluenceEstimator(mic_intervention_graph, num_nodes, 'act_prob_{}'.format(fake_component), obs_steps)
        exp_infl, std = influence_estimator.get_expected_influence_for_multiple_seed_sets(seed_sets, num_simulations)
        print('mic intervention: {} \pm {}'.format(exp_infl, std))

        """Run expected influence estimation (random edge intervention)
        """
        Z = np.random.choice(np.arange(len(base_df)), K, replace=False)
        discarded_edge_df = base_df.drop(Z[:K])
        random_intervention_graph = data_io.convert_dataframe_to_base_graph(discarded_edge_df, idx2u)
        
        influence_estimator = InfluenceEstimator(random_intervention_graph, num_nodes, 'act_prob_{}'.format(fake_component), obs_steps)
        exp_infl, std = influence_estimator.get_expected_influence_for_multiple_seed_sets(seed_sets, num_simulations)
        print('random intervention: {} \pm {}'.format(exp_infl, std))


    
    