#!/usr/bin/env python
# coding=utf-8
import snap
import numpy as np
import math
# np.random.seed(0)
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import time
import itertools
from multiprocessing import Pool, freeze_support

const = 10e-5
"""
Parameter estimation under discrete IC model (Saito). create_base_graph and train. it will add the "act_prob" inferred influnce as edge attribute. relaxation for discrete time as follows - consider memsize (window) of influencers. we do not choose to discretize time as that could mistakenly club wrong sets. edges u->v (if u can influnece v) or (v follows u). Pre-computations in create_base_graph [needed later in train: when creating u->v links with lookback limits]: "suvmi" and "suvpl" (subset of train_cascades contains S_{u,v}^{minus} as defined in Saito et al reqd for dQ=0 step).

create_base_graph and train:
-----------------------------
* Nodes/Edges selection [on train cascades]:
    - most active users (+extra users len so no cascade is removed),
    - edge threshold,
    - memsize/lookback_count (window of influence: in both functions since in the later edges
                can be a union from all cascades, not just the windowed influencers,
                reqd to match pre-computations step.) [computation time: neg_sampling removed]
"""

def _softmax(x):
    """
    numerically stable softmax computation
    """
    e_x = np.exp(x - np.max(x))
    softmax = e_x / e_x.sum()
    # argmax (binarize)
    # a = np.zeros(len(softmax))
    # np.put(a, np.argmax(softmax), 1)
    return softmax


def _metric(pred_train_labels, train_labels):
    # report clustering
    precision, recall, fscore, support = precision_recall_fscore_support(
        train_labels, pred_train_labels)
    acc = accuracy_score(train_labels, pred_train_labels)
    classification_report = pd.concat(
        map(pd.DataFrame, [[acc, acc], fscore, precision, recall, support]), axis=1)
    classification_report.columns = [
        "accuracy", "f1-score", "precision", "recall", "support"]
    print("Clustering results: Classification report")
    print(classification_report)
    # report flipped clustering
    precision, recall, fscore, support = precision_recall_fscore_support(
        train_labels, ~np.array(pred_train_labels, dtype=np.bool))
    flip_acc = accuracy_score(train_labels, ~np.array(pred_train_labels, dtype=np.bool))
    report_flip = pd.concat(map(pd.DataFrame, [[flip_acc, flip_acc], fscore, precision, recall, support]), axis=1)
    report_flip.columns = ["accuracy", "f1-score", "precision", "recall", "support"]
    print("Flipped prediction groups")
    print(report_flip)
    return max(acc, flip_acc)
    


def _update_act_prob(old_act_prob, denom_pl, denom_mi, numer_pl, component):
    """
    update old act prob according to m-step analytic solution.
    """
    d = denom_pl[component] + denom_mi[component]
    if d == 0: d = 10e-5
    new_act_prob = old_act_prob * numer_pl[component] * 1.0 / d
    if new_act_prob > 1: new_act_prob = 1.0
    elif new_act_prob < 0: new_act_prob = 0.0
    return new_act_prob


def _compute_ll(cascade, base_graph, lookback_count, index_dict, set_required_v_for_S, num_negative_samples):
    """
    Returns log-likelihood of given cascade under component IC model.
    """
    c0_ll, c1_ll = 0.0, 0.0
    num_nodes = base_graph.GetNodes()
    c0_prod_v_for_S = np.ones((num_nodes))
    c1_prod_v_for_S = np.ones((num_nodes))
    
    # Compute Eqn 6: Saito et al (p_nodeV_cascadeS)= 1 - Prod_influencers:u (1-P_u,v)
    # node iterator of node id = u (needs int not np.int)
    
    for u in index_dict:
        u_NI = base_graph.GetNI(int(u))  
        for v in u_NI.GetOutEdges():
            EI = base_graph.GetEI(int(u), int(v))
            attr0 = base_graph.GetFltAttrDatE(EI, "act_prob_0")
            attr1 = base_graph.GetFltAttrDatE(EI, "act_prob_1")
            if v in index_dict:
                # both are active in S, and we need u to be within lookback of v
                tu = index_dict[u]
                tv = index_dict[v]
                if tu <= tv - 1 and \
                    (lookback_count is None or tv <= tu + lookback_count):
                    c0_prod_v_for_S[v] *= 1 - attr0
                    c1_prod_v_for_S[v] *= 1 - attr1
            else:
                # v inactive in cascade, u active in cascade
                c0_prod_v_for_S[v] *= 1 - attr0
                c1_prod_v_for_S[v] *= 1 - attr1
    
    # sum up the values for log likelihood (over active nodes in S) except seed node
    active_users_in_S = cascade[1:, 0]
    c0_ll += np.log(const + 1 - c0_prod_v_for_S[active_users_in_S]).sum()
    c1_ll += np.log(const + 1 - c1_prod_v_for_S[active_users_in_S]).sum()
    
    # sum up the values for log likelihood (over inactive users given S)
    if num_negative_samples is not None:
        full_users = np.arange(num_nodes)
        inactive_users = set(full_users) - set(cascade[:, 0])
        chosen_inactive = np.random.choice(list(inactive_users), num_negative_samples, replace=False)
        mask = np.zeros((num_nodes), dtype=bool)
        mask[chosen_inactive] = True
        c0_ll += np.log(const + c0_prod_v_for_S[mask]).sum()
        c1_ll += np.log(const + c1_prod_v_for_S[mask]).sum()
    else:
        mask = np.ones((num_nodes), dtype=bool)
        mask[cascade[:, 0]] = False
        c0_ll += np.log(const + c0_prod_v_for_S[mask]).sum()
        c1_ll += np.log(const + c1_prod_v_for_S[mask]).sum()

    # Returning required pv_for_S
    if set_required_v_for_S:
        dict_computed_pv_for_S = {}
        for v in set_required_v_for_S:  # S stands for cascade sequence S
            dict_computed_pv_for_S[v] = (1 - c0_prod_v_for_S[v], 1 - c1_prod_v_for_S[v])
    else:
        dict_computed_pv_for_S = None
    return c0_ll, c1_ll, dict_computed_pv_for_S


def train(base_graph, train_cascades, train_labels, num_negative_samples=None, lookback_count=None,
          max_iter=100, freq_convergence_test=10):
    """
    Parameters
    ----------
    base_graph : PNEANet
        attributed graph with nodes (top active users) and potential edges (based on train_cascades).
        directed edges u->v exist (if u can influence v) or (v follows u).

    train_cascades : list(np.array(None, 2))
        training cascades i.e. list of [time ordered array of (user, time of activation)]

    num_negative_samples : int (default=None) [UNUSED]
        num of inactive users to sample (negative sampling). used to improve efficiency
        when computing log likelihood of all cascades under infered parameters from M-step.
        approximation of log likelihood used just for computational efficiency on large datasets.

    lookback_count : int (default=None)
        limit on number of potential influencers [i-lookback_count : i-1] of node i.
        required for continuous time handling relaxation.
        limit imposed for computational efficiency, and because closer users are more influential.

    lookback_timegap : [UNUSED]
        additional parameter to restrict potential influencers by time gap (besides count).
        time gap might not be useful if only top most active users are provided for inference.
        because then the gaps between them might be larger.

    max_iter : int (default=100)
        maximum iterations of EM for inference.

    freq_convergence_test : int (default=10) [UNUSED]
        frequency to test for convergence when log-likelihood worsens after next M-step.

    Returns
    --------
    float, float
        mixture weights inferred pi0, pi1.
    None
        PNEANet (snap package) same base_graph with attribute for edge influence weights updated.
        add new edge attr ("act_prob") if doesn't exist in graph for inferred influence.
    """
    # set index_dict_list to map nodeid to location/index in each cascade (fast access)
    index_dict_list = []
    for cascade in train_cascades:
        dict_ = {}
        users = cascade[:, 0]
        dict_ = dict(zip(users, np.arange(len(users))))
        index_dict_list.append(dict_)
    print('done setting index dict')

    # init random mixture wgts and edge act_probs for mixture IC model.
    pi0 = np.random.uniform(0.5, 0.5)
    pi1 = 1 - pi0
    base_graph.AddFltAttrE("act_prob_0")  # nothing happens if it already exists
    base_graph.AddFltAttrE("act_prob_1")  # nothing happens if it already exists
    # (cascade_ind: set(nodes v) for each we need pvS)
    required_pvS = [set() for _ in train_cascades]
    for EI in base_graph.Edges():        
        base_graph.AddFltAttrDatE(EI, np.random.uniform(0.1, 0.3), "act_prob_0")
        base_graph.AddFltAttrDatE(EI, np.random.uniform(0.1, 0.3), "act_prob_1")
        u, v = EI.GetSrcNId(), EI.GetDstNId()
        suvpl_str = base_graph.GetStrAttrDatE(EI, "suvpl")
        suvpl_cascades = suvpl_str.split(",")
        for cascade_ind_str in suvpl_cascades:
            if cascade_ind_str == '':
                continue
            cascade_ind = int(cascade_ind_str)
            required_pvS[cascade_ind].add(v)
        # for edges with no cascade in a component, make puv_comp = 0 from randominit
        
    print('done setting random initialization')
    # return pi0, pi1

    print("start: training")
    # best_graph = snap.ConvertGraph(type(base_graph), base_graph)  # copy graph (but it does not deepcopy)
    # best_acc = 0
    starttime = time.time()
    for step in range(max_iter):
        print("step = {} / {} in time till now = {:.3f}".format(step,
                                                                max_iter, time.time()-starttime))
        # E-step
        gamma0_list = np.zeros(len(train_cascades))
        gamma1_list = np.zeros(len(train_cascades))
        list_p_v_S = []  # list of dict_ (like required_pvS is a list of sets)
        # each index corresponds to train_cascades in order
        # set is for v (in pvS) and dict is for (v: (pvs_0, pvs_1))
        ll_cascades = 0.0
        for i, cascade in enumerate(train_cascades):
            ll_cascade_0, ll_cascade_1, dict_computed_pv_for_S = _compute_ll(
                cascade, base_graph, lookback_count, index_dict_list[i], required_pvS[i], num_negative_samples)
            list_p_v_S.append(dict_computed_pv_for_S)
            gamma0_d = math.log(const + pi0) + ll_cascade_0
            gamma1_d = math.log(const + pi1) + ll_cascade_1
            # print(ll_cascade_0, ll_cascade_1)
            ll_cascade = np.log(const + pi0 * np.exp(ll_cascade_0) + pi1 * np.exp(ll_cascade_1))  # ll_cascade_0 + ll_cascade_1
            ll_cascades += ll_cascade
            gamma0_list[i], gamma1_list[i] = _softmax([gamma0_d, gamma1_d])
        print('done E-step: update responsibilities gamma', ll_cascades)
        # print(gamma0_list)
        # print(gamma1_list)
        if step % 1 == 0:
            pred_train_labels = (gamma1_list >= 0.5)*1
            current_acc = _metric(pred_train_labels, train_labels)
            # if current_acc > best_acc: best_graph = snap.ConvertGraph(type(base_graph), base_graph)
        print('done evaluation of clustering accuracy at iter = {} at pi = [{}, {}]'.format(step, pi0, pi1))
     
        # M-step
        pi0 = np.mean(gamma0_list)
        pi1 = 1 - pi0
        for EI in base_graph.Edges():

            denom_pl, denom_mi, numer_pl = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]

            # go over the suvpl cascade subset
            v = EI.GetDstNId()
            suvpl_str = base_graph.GetStrAttrDatE(EI, "suvpl")
            suvpl_cascades = suvpl_str.split(",")
            for cascade_ind_str in suvpl_cascades:
                if cascade_ind_str == '':
                    continue
                cascade_ind = int(cascade_ind_str)
                denom_pl[0] += gamma0_list[cascade_ind]
                denom_pl[1] += gamma1_list[cascade_ind]

                # calculate numer_pl of component 0
                pvs_0 = list_p_v_S[cascade_ind][v][0]
                if pvs_0 == 0: pvs_0 = 10e-5
                numer_pl[0] += gamma0_list[cascade_ind] * 1.0 / pvs_0

                # calculate numer_pl of component 1
                pvs_1 = list_p_v_S[cascade_ind][v][1]
                if pvs_1 == 0: pvs_1 = 10e-5
                numer_pl[1] += gamma1_list[cascade_ind] * 1.0 / pvs_1

            # go over the suvmi cascade subset
            suvmi_str = base_graph.GetStrAttrDatE(EI, "suvmi")
            suvmi_cascades = suvmi_str.split(",")
            for cascade_ind_str in suvmi_cascades:
                if cascade_ind_str == '':
                    continue
                cascade_ind = int(cascade_ind_str)
                denom_mi[0] += gamma0_list[cascade_ind]
                denom_mi[1] += gamma1_list[cascade_ind]

            # update the act_prob_0 and act_prob_1
            old_act_prob_0 = base_graph.GetFltAttrDatE(EI, "act_prob_0")
            new_act_prob_0 = _update_act_prob(
                old_act_prob_0, denom_pl, denom_mi, numer_pl, 0)
            base_graph.AddFltAttrDatE(EI, new_act_prob_0, "act_prob_0")

            old_act_prob_1 = base_graph.GetFltAttrDatE(EI, "act_prob_1")
            new_act_prob_1 = _update_act_prob(
                old_act_prob_1, denom_pl, denom_mi, numer_pl, 1)
            base_graph.AddFltAttrDatE(EI, new_act_prob_1, "act_prob_1")           

    print("done: training")

    return pi0, pi1


def last_evaluation(pi0, pi1, base_graph, train_cascades, train_labels, num_negative_samples=None, lookback_count=None):
    """
    Updating the responsibilities and clustering after last M-step.
    """
    # set index_dict_list to map nodeid to location/index in each cascade (fast access)
    index_dict_list = []
    for cascade in train_cascades:
        dict_ = {}
        users = cascade[:, 0]
        dict_ = dict(zip(users, np.arange(len(users))))
        index_dict_list.append(dict_)
    print('done setting index dict')

    # E-step
    ll_cascades = 0
    gamma0_list = np.zeros(len(train_cascades))
    gamma1_list = np.zeros(len(train_cascades))
    for i, cascade in enumerate(train_cascades):
        ll_cascade_0, ll_cascade_1, _ = _compute_ll(
            cascade, base_graph, lookback_count, index_dict_list[i], None, num_negative_samples)
        gamma0_d = math.log(const + pi0) + ll_cascade_0
        gamma1_d = math.log(const + pi1) + ll_cascade_1
        ll_cascade = np.log(const + pi0 * np.exp(ll_cascade_0) + pi1 * np.exp(ll_cascade_1))  # ll_cascade_0 + ll_cascade_1
        ll_cascades += ll_cascade
        gamma0_list[i], gamma1_list[i] = _softmax([gamma0_d, gamma1_d])
    print('done: recompute responsibilities gamma')

    pred_train_labels = (gamma1_list >= 0.5)*1
    _metric(pred_train_labels, train_labels)
    print('done evaluation of clustering accuracy at end at pi = [{}, {}], ll={}, {}.'.format(pi0, pi1, ll_cascades, ll_cascades/len(train_cascades)))
    return np.array(gamma0_list), np.array(gamma1_list), np.array(train_labels)



#     # unoptimized version
#     for EI in base_graph.Edges():
#         u = EI.GetSrcNId()
#         v = EI.GetDstNId()
        
#         attr0 = base_graph.GetFltAttrDatE(EI, "act_prob_0")
#         attr1 = base_graph.GetFltAttrDatE(EI, "act_prob_1")
        
#         if v in index_dict and u in index_dict:
#             # both are active in S, and we need u to be within lookback of v
#             tu = index_dict[u]
#             tv = index_dict[v]
#             if tu <= tv - 1 and \
#                     (lookback_count is None or tv <= tu + lookback_count):
#                 c0_prod_v_for_S[v] *= 1 - attr0
#                 c1_prod_v_for_S[v] *= 1 - attr1
#             print('here', u, v, u in index_dict, v in index_dict)

#         if v not in index_dict and u in index_dict:
#             # v inactive in cascade, u active in cascade
#             c0_prod_v_for_S[v] *= 1 - attr0
#             c1_prod_v_for_S[v] *= 1 - attr1
#             print('here too', u, v, u in index_dict, v in index_dict)