# -*- coding: utf-8 -*-
import numpy as np
import sys, os, time
sys.path.append("../")
from simulator.greedy_clef import select_seeds_greedy_clef, InfluenceEstimator
import snap
import pandas as pd
import pickle
from parameter_estimation import data_io


if __name__ == '__main__':
    start = time.time()
    """Input setting
    --------------------
    """
    data = 'all_kwon/kwon_copy'
    
    pi_file = '../output/{}/pi.txt'.format(data)
    edges_file = '../output/{}/learned_graph.tsv'.format(data)
    idx2u_file = '../output/{}/idx2u.txt'.format(data)
    
    save_influential_users_file = '../output/{}/selected_influential_users.tsv'.format(data)
    
    log_file = '../output/{}/infmax_log.txt'.format(data)
    
    # number of most influential users to select with greedy algorithm
    K = 20
    num_simulations = 10
    obs_steps = 10
    
    logf = open(log_file, 'w')
    sys.stdout = data_io.Runlog(sys.stdout, logf)  # This will go to stdout and the file out.txt
    
    print('K = {}'.format(K))
    print('num_simulations = {}'.format(num_simulations))
    print('obs_steps = {}'.format(obs_steps))
    
    """Load graph and activation probabilities (learned and saved from parameter estimation)
    --------------------
    """
    pi0, pi1, base_graph, idx2u, u2idx = data_io.load_estimated_parameters(pi_file, edges_file, idx2u_file)
    num_nodes = base_graph.GetNodes()
    assert num_nodes == len(idx2u), 'inconsistent num of nodes in base_graph and in idx2u'
    
    comp0_influence_estimator = InfluenceEstimator(base_graph, num_nodes, 'act_prob_0', obs_steps)
    comp1_influence_estimator = InfluenceEstimator(base_graph, num_nodes, 'act_prob_1', obs_steps)
    
    """
    select most influential seeds under component 0
    """
    selected_0, influence_0 = select_seeds_greedy_clef(K, num_nodes, comp0_influence_estimator, num_simulations)
    print("component 0:", selected_0, influence_0)
    

    """
    select most influential seeds under component 1
    """
    selected_1, influence_1 = select_seeds_greedy_clef(K, num_nodes, comp1_influence_estimator, num_simulations)
    print("component 1:", selected_1, influence_1)
    
    """
    saving inferred influential seeds for both components
    """
    data_io.save_selected_influential_users(selected_0, influence_0, selected_1, influence_1, save_influential_users_file)
    
    print('=> Note that comp0 and comp1 may correspond to true/fake (mapping is known by cross-validation)')
    print('finished program.')