# -*- coding: utf-8 -*-
import heapq
import time
import sys
sys.path.append("../")
from simulator import dic_simulator
import numpy as np

class InfluenceEstimator(object):

    def __init__(self, graph, num_nodes, act_prob_constant, obs_steps):
        self.graph = graph
        self.num_nodes = num_nodes
        self.act_prob_constant = act_prob_constant
        self.obs_steps = obs_steps

    def get_expected_influence(self, seedset, num_simulations):
        cascades = dic_simulator.simulate(seedset, num_simulations, self.graph, self.num_nodes,
            self.act_prob_constant, self.obs_steps)
        total_len = 0
        for cascade in cascades:
            total_len += len(cascade)
        fs = total_len/len(cascades)
        del cascades
        return fs
    
    
    def get_expected_influence_per_timestep(self, seed_set, num_simulations):
        cascades = dic_simulator.simulate(seedset, num_simulations, self.graph, self.num_nodes,
            self.act_prob_constant, self.obs_steps)
        
        maxlen = 0
        for cascade in cascades:
            m = np.max(cascade[:, 1])
            maxlen = max(maxlen, m)
            
        engagements = np.zeros((len(cascades), maxlen))
        for i, cascade in enumerate(cascades):
            unique, counts = np.unique(cascade[:,1], return_counts=True)
            engagements[i, unique] = counts
        
        expected_cascade = np.mean(engagements, 0)
        std = np.std(engagements, 0)
        del enagagements, cascades
        # cum = 1.0* np.cumsum(mean)
        # cumnorm = cum/np.max(cum)
        # print("Final expected size of cascade:", np.max(cum))
        # print("Max # tweets expected at any timestep:", np.max(mean))
        return expected_cascade, std  # exp_per_time, std_per_time
        
        
    def get_expected_influence_for_multiple_seed_sets(self, seed_sets, num_simulations):
        generated = []
        for i, seedset in enumerate(seed_sets):
            if i % 10 == 0:
                print('generating cascades for {}/{}th seed set'.format(i, len(seed_sets)))
            cascades = dic_simulator.simulate(seedset, num_simulations, self.graph, self.num_nodes, 
                                    self.act_prob_constant, self.obs_steps)
            generated += cascades
        cas_lengths = np.array([len(cascade) for cascade in generated])
        m, s = np.mean(cas_lengths), np.std(cas_lengths)
        del cascades, generated, cas_lengths
        return m, s
    
    
class HeapObj(object):

    def __init__(self, node_id, marg_gain):
        self.marg_gain = marg_gain
        self.node_id = node_id

    """ reversed comparisons (high is low): to use min-heap implementation from heapq """
    def __cmp__(self, other):
        return -cmp(self.marg_gain, other.marg_gain)

    def __gt__(self, other):
        return self.marg_gain < other.marg_gain

    def __lt__(self, other):
        return self.marg_gain > other.marg_gain

    def __repr__(self):
        return ("( %d, %0.2f )" % (self.node_id, self.marg_gain))

def select_seeds_greedy_clef(k, num_nodes, influence_estimator, num_sims):
    """ for fast montone, submodular set function optimization (max f(S) subject to cardinality constraint k) """
    print("Begin greedy seed selection:")
    start_time = time.time()
    selected_seeds = list()
    """ input: influence_estimator computes f(S) = expected number of nodes activated under seedset S """
    marg_gain_heap = [HeapObj(node, influence_estimator.get_expected_influence(
        seedset=[node], num_simulations=num_sims)) for node in range(num_nodes)]
    print("running heapify")
    heapq.heapify(marg_gain_heap)
    print("done heapify")
    # print(marg_gain_heap)
    first_selected_seed = heapq.heappop(marg_gain_heap)
    selected_seeds.append(first_selected_seed.node_id)
    influence_selected_seeds = first_selected_seed.marg_gain
    print("selected 0", first_selected_seed)

    for iteration in range(k-1):

        heap_change = True

        while heap_change:
            # print("In", iteration, len(marg_gain_heap), counter, visited_node_set)
            # Recalculate spread of top node
            current_top_node_obj = heapq.heappop(marg_gain_heap)
            current_top_node = current_top_node_obj.node_id
            updated_marg_gain = influence_estimator.get_expected_influence(
                seedset=selected_seeds+[current_top_node], num_simulations=num_sims) - influence_selected_seeds
            # print(updated_marg_gain, current_top_node, len(marg_gain_heap))
            current_top_node_obj.marg_gain = updated_marg_gain
            heapq.heappush(marg_gain_heap, current_top_node_obj)
            heap_change = (marg_gain_heap[0].node_id != current_top_node and 
                           marg_gain_heap[0].marg_gain != updated_marg_gain)

        # Select the best node as computed from above loop
        selected = heapq.heappop(marg_gain_heap)
        selected_seeds.append(selected.node_id)
        influence_selected_seeds += selected.marg_gain
        print("selected %d " % (iteration+1), selected)

    print("End greedy seed selection in %0.2f" % (time.time() - start_time))
    del marg_gain_heap, first_selected_seed, current_top_node_obj
    return selected_seeds, influence_selected_seeds
