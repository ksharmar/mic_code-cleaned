#!/usr/bin/env python
# coding=utf-8
import snap
import numpy as np
# np.random.seed(0)
const = 10e-5

def train(base_graph, train_cascades, num_negative_samples=None, lookback_count=None, \
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
        required for continuos time handling relaxation.
        limit imposed for computational efficiency, and because closer users are more influential.

    lookback_timegap : not implemented
        additional parameter to restrict potential influencers by time gap (besides count).
        time gap might not be useful if only top most active users are provided for inference.
        because then the gaps between them might be larger.

    max_iter : int (default=100)
        maximum iterations of EM for inference.

    freq_convergence_test : int (default=10) [UNUSED]
        frequency to test for convergence when log-likelihood worsens after next M-step.

    Returns
    --------
    None
        PNEANet (snap package) same base_graph with attribute for edge influence weights updated.
        add new edge attr ("act_prob") if doesn't exist in graph for inferred influence.
    """
    # set index_dict_list to map nodeid to location/index in each cascade (fast access)
    index_dict_list = _set_index_dict(train_cascades)
    # set random act_prob to init EM inference.
    _init_train(base_graph)
    print("start: training")
    for step in range(max_iter):
        print("step = {} / {}".format(step, max_iter))
        # E and M-step
        for EI in base_graph.Edges():
            v = EI.GetDstNId()
            suvpl_str = base_graph.GetStrAttrDatE(EI, "suvpl")
            suvpl_cascades = suvpl_str.split(",")
            temp_sum = 0.0
            for cascade_ind_str in suvpl_cascades:
                if cascade_ind_str == '': continue
                cascade_ind = int(cascade_ind_str)
                temp_sum += 1.0 / _computePv(v, train_cascades[cascade_ind], base_graph,
                    lookback_count, index_dict_list[cascade_ind])
            ratio = base_graph.GetFltAttrDatE(EI, "ratio")
            act_prob = base_graph.GetFltAttrDatE(EI, "act_prob_0")
            updated_act_prob = ratio * act_prob * temp_sum
            assert(updated_act_prob >= 0.0 and updated_act_prob <= 1.0)
            base_graph.AddFltAttrDatE(EI, updated_act_prob, "act_prob_0")
        last_evaluation(base_graph, train_cascades, num_negative_samples, lookback_count)
    print("done: training")


def _set_index_dict(train_cascades):
    """
    set list of dict (corresponding to each train_cascade)
    dict stores mapping from nodeid to index (location) in cascade
    """
    index_dict_list = []
    for cascade in train_cascades:
        dict_ = {}
        for ii, u in enumerate(cascade[:, 0]):
            dict_[u] = ii
        index_dict_list.append(dict_)
    return index_dict_list


def _init_train(base_graph):
    """
    Init for EM: Set random activation probabilties
    and the ratio which is used in analytic form.
    """
    base_graph.AddFltAttrE("act_prob_0")  # nothing happens if it already exists
    base_graph.AddFltAttrE("ratio")

    # set random act_prob and fixed ratio used in analytical formula
    for EI in base_graph.Edges():
        suvmi_str = base_graph.GetStrAttrDatE(EI, "suvmi")
        suvpl_str = base_graph.GetStrAttrDatE(EI, "suvpl")
        suvmi_len = 0 if suvmi_str == '' else len(suvmi_str.split(","))
        suvpl_len = 0 if suvpl_str == '' else len(suvpl_str.split(","))
        # assert suvmi_len + suvpl_len > 0, 'length of suvmi + suvpl should be atleast 1'
        # TODO get back assertion and remove 10e-5 print(suvmi_len + suvpl_len)
        r = 1.0 / (10e-5 + suvmi_len + suvpl_len)
        base_graph.AddFltAttrDatE(EI, r, "ratio")
        base_graph.AddFltAttrDatE(EI, np.random.rand(), "act_prob_0")


def _computePv(v, cascade, base_graph, lookback_count, index_dict):
    """
    Compute Eqn 6: Saito et al (p_nodeV_cascadeS)= 1 - Prod influencers=u (1-P_u,v)
    """
    # Find ulist (u in parents(v) & A_s_previous) from cascade s
    # index_dict stores index of v in cascade
    end_index = index_dict[v]
    start_index = 0 if lookback_count is None else end_index - lookback_count
    if start_index < 0:
        start_index = 0
    ulist = cascade[start_index:end_index, 0]
    assert len(ulist) > 0
    prod = 1.0
    for u in ulist:
        u = int(u); v = int(v)
        if not base_graph.IsEdge(u, v):
            continue
        EId = base_graph.GetEId(u, v)  # returns an int edge id
        act_prob_uv =  base_graph.GetFltAttrDatE(EId, "act_prob_0")
        prod *= (1.0 - act_prob_uv)
    return 1.0 - prod  # +10e-5








def _compute_ll(cascade, base_graph, lookback_count, index_dict, set_required_v_for_S, num_negative_samples):
    """
    Returns log-likelihood of given cascade under component IC model.
    """
    c0_ll = 0.0
    num_nodes = base_graph.GetNodes()
    c0_prod_v_for_S = np.ones((num_nodes))
    
    # Compute Eqn 6: Saito et al (p_nodeV_cascadeS)= 1 - Prod_influencers:u (1-P_u,v)
    # node iterator of node id = u (needs int not np.int)
    
    for u in index_dict:
        u_NI = base_graph.GetNI(int(u))  
        for v in u_NI.GetOutEdges():
            EI = base_graph.GetEI(int(u), int(v))
            attr0 = base_graph.GetFltAttrDatE(EI, "act_prob_0")
            if v in index_dict:
                # both are active in S, and we need u to be within lookback of v
                tu = index_dict[u]
                tv = index_dict[v]
                if tu <= tv - 1 and \
                    (lookback_count is None or tv <= tu + lookback_count):
                    c0_prod_v_for_S[v] *= 1 - attr0
            else:
                # v inactive in cascade, u active in cascade
                c0_prod_v_for_S[v] *= 1 - attr0
    
    # sum up the values for log likelihood (over active nodes in S) except seed node
    active_users_in_S = cascade[1:, 0]
    c0_ll += np.log(const + 1 - c0_prod_v_for_S[active_users_in_S]).sum()
    
    # sum up the values for log likelihood (over inactive users given S)
    if num_negative_samples is not None:
        full_users = np.arange(num_nodes)
        inactive_users = set(full_users) - set(cascade[:, 0])
        chosen_inactive = np.random.choice(list(inactive_users), num_negative_samples, replace=False)
        mask = np.zeros((num_nodes), dtype=bool)
        mask[chosen_inactive] = True
        c0_ll += np.log(const + c0_prod_v_for_S[mask]).sum()
    else:
        mask = np.ones((num_nodes), dtype=bool)
        mask[cascade[:, 0]] = False
        c0_ll += np.log(const + c0_prod_v_for_S[mask]).sum()

    # Returning required pv_for_S
    if set_required_v_for_S:
        dict_computed_pv_for_S = {}
        for v in set_required_v_for_S:  # S stands for cascade sequence S
            dict_computed_pv_for_S[v] = 1 - c0_prod_v_for_S[v]
    else:
        dict_computed_pv_for_S = None
    return c0_ll, dict_computed_pv_for_S

def last_evaluation(base_graph, train_cascades, num_negative_samples=None, lookback_count=None):
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
    for i, cascade in enumerate(train_cascades):
        ll_cascade_0, _ = _compute_ll(
            cascade, base_graph, lookback_count, index_dict_list[i], None, num_negative_samples)
        ll_cascades += ll_cascade_0
    print('done evaluation of clustering accuracy at end ll={}.'.format(ll_cascades))