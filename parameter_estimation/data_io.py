# after parameter estimation we need to save the learned parameters (graph)
import pandas as pd
import numpy as np
import snap

"""
Results saved in
pi (np.txt) [pi0, pi1]
edge_file (TSV) [u, v, act_0, act_1] (indices, not ids)
id2u (np.txt) list of user ids corresponding to index 0 to num_users, in order.
influential_users_file (TSV) [K, selected_0, selected_1, fs0, fs1]
"""


class Runlog(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()
            

def save_estimated_parameters(pi, base_graph, idx2u, save_pi_file, save_edges_file, save_idx2u_file):
    """
    SAVE PI
    """
    np.savetxt(save_pi_file, pi)
    print('saved pi0, pi1 at {}'.format(save_pi_file))

    """
    SAVE EDGES AND ATTRIBUTES as TSV(u, v, act_0, act_1)  indices not ids
    """
    act_0, act_1, list_u, list_v = [], [], [], []
    for EI in base_graph.Edges():
        u, v = EI.GetSrcNId(), EI.GetDstNId() 
        p0 = base_graph.GetFltAttrDatE(EI, "act_prob_0")
        p1 = base_graph.GetFltAttrDatE(EI, "act_prob_1")
        if p0 == 0 and p1 == 0: continue
        list_u.append(u); list_v.append(v)
        act_0.append(p0); act_1.append(p1)

    df = pd.DataFrame({'u': list_u, 'v': list_v, 'act_0': act_0, 'act_1': act_1})
    # df['u_id'] = df['u_index'].apply(lambda i: idx2u[i])
    # df['v_id'] = df['v_index'].apply(lambda i: idx2u[i])

    df.to_csv(save_edges_file, index=False, sep='\t')
    print("saved_graph at location: {}".format(save_edges_file))

    """
    SAVE IDX2U
    """
    np.savetxt(save_idx2u_file, idx2u)
    print("saved_idx2u at location: {}".format(save_idx2u_file))


def save_selected_influential_users(selected_0, influence_0, selected_1, influence_1, save_influential_users_file):
    """
    SAVE INFL USERS as TSV(K, selected_0, selected_1, fs0, fs1)
    """
    assert len(selected_0) == len(
        selected_1), 'unequal K in greedy selection for the components.'
    length = len(selected_0)
    fs0, fs1 = [influence_0] * length, [influence_1] * length
    df = pd.DataFrame({'K': np.arange(length), 'selected_0': selected_0,
                       'selected_1': selected_1, 'fs0': fs0, 'fs1': fs1})
    df.to_csv(save_influential_users_file, sep='\t', index=False)

    print('saved selected influential users for both components at {}'.format(
        save_influential_users_file))


def load_estimated_parameters(pi_file, edges_file, idx2u_file):
    pi0, pi1 = np.loadtxt(pi_file)
    idx2u = np.loadtxt(idx2u_file)
    u2idx = dict(zip(idx2u, np.arange(len(idx2u))))
    base_df = pd.read_csv(edges_file, header=0, sep='\t')
    base_graph = convert_dataframe_to_base_graph(base_df, idx2u)
    return pi0, pi1, base_graph, idx2u, u2idx


# def load_graph_from_tsv(edge_file, idx2u):
#     base_df = pd.read_csv(edge_file, header=0, sep='\t')
#     graph = snap.LoadEdgeList(snap.PNEANet, base_df, 0, 1)
#     # check, else remove headers
#     for i in range(len(idx2u)):
#         if graph.IsNode(i): continue
#         graph.AddNode(i)
#     assert graph.GetNodes() == len(idx2u), 'inconsistent num of nodes'


#     act_0 = base_df['act_0']
#     act_1 = base_df['act_1']
#     for i, EI in enumerate(graph.Edges()):
#         u, v = EI.GetSrcNId(), EI.GetDstNId()
#         # check if edges stored in order of act probs
#         graph.AddFltAttrDatE(EI, act_0[i], 'act_prob_0')
#         graph.AddFltAttrDatE(EI, act_1[i], 'act_prob_1')
#     return graph


def convert_dataframe_to_base_graph(base_df, idx2u):
    """TSV(u, v, act_0, act_1)
    Return snap.PNEANet
    """
    graph = snap.PNEANet.New()
    for i in range(len(idx2u)):
        graph.AddNode(i)

    for i, row in base_df.iterrows():
        u, v, act0, act1 = int(row['u']), int(row['v']), float(row['act_0']), float(row['act_1'])
        graph.AddEdge(u, v)
        EId = graph.GetEId(u, v)
        graph.AddFltAttrDatE(EId, act0, 'act_prob_0')
        graph.AddFltAttrDatE(EId, act1, 'act_prob_1')

    print('graph created:', graph.GetNodes(), graph.GetEdges())
    return graph


def load_base_graph_as_dataframe(edges_file):
    # TSV(u, v, act_0, act_1)
    return pd.read_csv(edges_file, header=0, sep='\t')


def get_act_prob_from_dataframe(base_df, component):
    act_prob_comp = base_df['act_{}'.format(component)]
    return np.array(act_prob_comp)


def load_selected_infl_users(infl_users_file, component):
    # TSV(K, selected_0, selected_1, fs0, fs1)
    infl_df = pd.read_csv(infl_users_file, header=0, sep='\t')
    selected_in_comp = infl_df['selected_{}'.format(component)]
    return selected_in_comp


def load_base_graph_from_files(num_nodes, edge_file, edge_file_header, act_0_file, act_1_file):

    act_0, act_1 = np.loadtxt(act_0_file), np.loadtxt(act_1_file)
    df = pd.read_csv(edge_file, sep='\t', header=edge_file_header)

    graph = snap.PNEANet.New()
    for i in range(num_nodes):
        graph.AddNode(i)

    for i, row in df.iterrows():
        u, v = int(row[0]), int(row[1])
        graph.AddEdge(u, v)
        EId = graph.GetEId(u, v)
        graph.AddFltAttrDatE(EId, act_0[i], 'act_prob_0')
        graph.AddFltAttrDatE(EId, act_1[i], 'act_prob_1')

    print('graph created:', graph.GetNodes(), graph.GetEdges())
    return graph
