import snap
import numpy as np
from collections import defaultdict
import operator
from scipy import sparse
# np.random.seed(0)


def _read_cascades_file(cascades_filename):
    """
    Returns
    -------
    cascades : list(np.array((None, 2)))
        list of user_str, timestamp array (one array per cascade)
    """
    f = open(cascades_filename, "r")
    cascades = []
    for line in f.readlines():
        u_t = line.strip("\n").split(",")
        u = list(map(int, u_t[0::2]))  # int
        t = np.array(list(map(float, u_t[1::2])), dtype=np.int32)  # float
        cascade = np.vstack([u, t]).transpose()
        cascades.append(cascade)
    f.close()
    return cascades   
    

def _read_train_cascade_ids_file(train_cascade_ids_filename):
    """
    Returns
    -------
    train_ids : np.array(int)
    """
    train_ids = np.loadtxt(train_cascade_ids_filename, dtype=np.int32)
    return train_ids


def _read_labels_file(labels_filename):
    """
    Returns
    -------
    train_labels : np.array(int)
    """
    cascade_labels = np.loadtxt(labels_filename, dtype=np.int32)
    return cascade_labels


def _convert_cascades_by_index(raw_observed_cascades, u2idx):
    """
    Returns
    -------
    cascades : list(np.array((None, 2)))
        list of user_idx, timestamp array (one array per cascade) as a np.array
    """
    indexed_cascades = []
    counter = 0
    for cascade in raw_observed_cascades:
        filtered = cascade[np.isin(cascade[:,0], list(u2idx.keys()))] 
        if len(filtered) == 0:
            counter += 1
            indexed_cascades.append([])
        else:
            filtered[:,0] = np.vectorize(u2idx.get)(filtered[:,0])
            indexed_cascades.append(filtered)
    print('num_empty_cascades', counter)
    return np.array(indexed_cascades)
#     for cascade in raw_observed_cascades: 
#         u_ids = []
#         t = []
#         for u_str, timestamp in zip(cascade[:, 0], cascade[:, 1]):
#             if u_str not in u2idx:
#                 continue
#             u_ids.append(u2idx[u_str])
#             t.append(int(float(timestamp)))
#         u_ids = np.array(u_ids)
#         t = np.array(t)
#         indexed_cascades.append(np.vstack([u_ids, t]).transpose())
#     return np.array(indexed_cascades)


def _build_user_index(raw_observed_cascades, user_max, extra_users_len):
    """
    Returns
    -------
    u2idx, idx2u
    """
    # calculate activity_count by appearance of users in cascades
    activity_count = defaultdict(int)
    for cascade in raw_observed_cascades:
        u_str_list = cascade[:, 0]
        for u_str in u_str_list:
            activity_count[u_str] = activity_count[u_str] + 1
    # retain user_max users with highest activity count
    resorted_users = sorted(activity_count.items(), key=operator.itemgetter(1), reverse=True)
    retained_user_np = np.array(resorted_users[0:user_max])[:,0]
    retained_set = set(retained_user_np)
    # add first "extra_users_len" users into the index
    if extra_users_len is not None and extra_users_len > 0:
        for cascade in raw_observed_cascades:
            u_str_first_list = cascade[:extra_users_len, 0]
            retained_set.update(u_str_first_list)
    # create index mapping
    u2idx = {}
    idx2u = []
    for ii, u_str in enumerate(retained_set):
        u2idx[u_str] = ii
        idx2u.append(u_str)
    return u2idx, idx2u


def _create_base_graph(train_cascades, idx2u, lookback_count=None, edge_thr=5):
    """
    Returns
    -------
    base_graph : PNEANet
        nodes, u influences v edges with lookback_count, suvmi and suvpl sets of train cascade indices
    """
    base_graph = snap.PNEANet.New()
    # add nodes
    for i in range(len(idx2u)):
        base_graph.AddNode(i)
    print("added nodes", base_graph.GetNodes())

    # set index_dict_list to map nodeid to location/index in each cascade (fast access)
    # i.e list of dict that contains mapping from nodeid to index (location) in cascade
    index_dict_list = []  # list of dict (corresponding to each train_cascade)
    user_participation = defaultdict(set)
    # add directed edges (u potentially influences v)
#     list_edges = sparse.lil_matrix((len(idx2u), len(idx2u)), dtype=int)  # base_graph.AddIntAttrE("edge_counts")
    for train_cascade_ind, cascade in enumerate(train_cascades):
        if train_cascade_ind % 100 == 0:
            print("processed {} / {}".format(train_cascade_ind, len(train_cascades)))
        dict_ = {}
        users = cascade[:, 0]
        for i, u in enumerate(users):
            dict_[u] = i
            user_participation[u].add(train_cascade_ind)
            v_users = users[i + 1 : i + lookback_count + 1] if lookback_count is not None else users[i + 1 :]
#             rows = np.array([u]).reshape(-1, 1)  # u_users = np.repeat(u, len(v_users), 0)
#             list_edges[rows, v_users] += np.ones((rows.size, v_users.size))           
            for v in v_users:
                # if str(u) + "-" + str(v) not in edge_counts:
                #     edge_counts[str(u) + "-" + str(v)] = 0
                # edge_counts[str(u) + "-" + str(v)] = edge_counts[str(u) + "-" + str(v)] + 1
                u, v = int(u), int(v)
                if not base_graph.IsEdge(u, v):
                    base_graph.AddEdge(u, v)
                    ei = base_graph.GetEI(u, v)
                    base_graph.AddIntAttrDatE(ei, 1, "edge_counts")
                else:
                    ei = base_graph.GetEI(u, v)
                    c = base_graph.GetIntAttrDatE(ei, "edge_counts")
                    base_graph.AddIntAttrDatE(ei, c + 1, "edge_counts")
        index_dict_list.append(dict_)

    print("added all potential edges")

#    list_edges = list_edges.multiply(list_edges >= edge_thr)
    remove_edge_list = []
    for i, EI in enumerate(base_graph.Edges()):
        # print(EI.GetSrcNId(), EI.GetDstNId())
        c = base_graph.GetIntAttrDatE(EI, "edge_counts")
        if c < edge_thr:
            remove_edge_list.append(EI.GetId())
    for edge_id in remove_edge_list:
        # print(EI.GetSrcNId(), EI.GetDstNId())
        # if base_graph.IsEdge(EI):
        base_graph.DelEdge(edge_id) # .GetSrcNId(), EI.GetDstNId())
    print("removed edges below thr")
    
#     # update cascade counts for diffusion edges
#     # suvpl (A_uv) is set of cascades where u is active and v is active at next timestep after u's activation
#     # ts (v) = ts (u) + 1 (which in our case means within lookback_count)
#     # suvmi (B_uv) is the number of cascades u is activated at some time ts(u) and v is not
#     # activated up to and including time ts(u) + 1 (which in our case means within lookback_count)
#     # i.e ts(v) > ts(u) + 1 or inf
#     Ks = list_edges.tocsr().nonzero()
#     e = np.vstack([Ks[0], Ks[1]]).transpose()
#     for i, (u, v) in enumerate(e):
#         u, v = int(u), int(v)  # numpy to regular int (otherwise AddEdge won't work)
#         if i % 1000 == 0:
#             print("set edge attributes {} / {}".format(i, len(e)))   
#         base_graph.AddEdge(u, v)
#         EI = base_graph.GetEI(u, v)
#         suvpl_set = set()
#         suvmi_set = set()
        
#         # cascade indices in which u is actived
#         for cascade_ind in user_participation[u]:
#             index_dict = index_dict_list[cascade_ind]
#             # add to suvpl_set
#             if v in index_dict and index_dict[v] >= index_dict[u] + 1 and \
#                 (lookback_count is None or index_dict[v] <= index_dict[u] + lookback_count):
#                 suvpl_set.add(cascade_ind)
#             # add to suvmi_set
#             if lookback_count is None and v not in index_dict:
#                 # v must not be active in the cascade at all
#                 suvmi_set.add(cascade_ind)
#             elif lookback_count is not None and (v not in index_dict 
#                                                  or index_dict[v] > index_dict[u] + lookback_count):
#                 # v must not be active upto and including lookback_count (inactive or late active)
#                 suvmi_set.add(cascade_ind)

#         # set as edge attributes
#         suvpl_str = ",".join(map(str, suvpl_set))
#         base_graph.AddStrAttrDatE(EI, suvpl_str, "suvpl")
#         suvmi_str = ",".join(map(str, suvmi_set))
#         base_graph.AddStrAttrDatE(EI, suvmi_str, "suvmi")
 
#     print("add edges to base graph")
    
    # update cascade counts for diffusion edges
    # suvpl (A_uv) is set of cascades where u is active and v is active at next timestep after u's activation
    # ts (v) = ts (u) + 1 (which in our case means within lookback_count)
    # suvmi (B_uv) is the number of cascades u is activated at some time ts(u) and v is not
    # activated up to and including time ts(u) + 1 (which in our case means within lookback_count)
    # i.e ts(v) > ts(u) + 1 or inf
    for i, EI in enumerate(base_graph.Edges()):
        if i % 5000 == 0:
            print("set edge attributes {} / {}".format(i, base_graph.GetEdges()))
        u, v = EI.GetSrcNId(), EI.GetDstNId()
        suvpl_set = set()
        suvmi_set = set()

        # cascade indices in which u is actived
        for cascade_ind in user_participation[u]:
            index_dict = index_dict_list[cascade_ind]
            # add to suvpl_set
            if v in index_dict and index_dict[v] >= index_dict[u] + 1 and \
                (lookback_count is None or index_dict[v] <= index_dict[u] + lookback_count):
                suvpl_set.add(cascade_ind)
            # add to suvmi_set
            if lookback_count is None and v not in index_dict:
                # v must not be active in the cascade at all
                suvmi_set.add(cascade_ind)
            elif lookback_count is not None and (v not in index_dict 
                                                 or index_dict[v] > index_dict[u] + lookback_count):
                # v must not be active upto and including lookback_count (inactive or late active)
                suvmi_set.add(cascade_ind)

        # set as edge attributes
        suvpl_str = ",".join(map(str, suvpl_set))
        base_graph.AddStrAttrDatE(EI, suvpl_str, "suvpl")
        suvmi_str = ",".join(map(str, suvmi_set))
        base_graph.AddStrAttrDatE(EI, suvmi_str, "suvmi")

    return base_graph


def load_data_for_parameter_estimation(cascades_filename, labels_filename,
    train_cascade_ids_filename, user_max, extra_users_len, lookback_count,
    edge_thr, cascade_count, min_cas_length):
    """
    Load raw dataset, prune cascades and build index by most active users (and extra_users_len),
    build indexed train_cascades and val_cascades (by filtering on input train cascade ids),
    create base_graph with lookback_count (if u potentially infuences v in any train cascade).

    Parameters
    ----------
    cascades_filename : string
        full path of file with observed cascades (format => each line corresponds to one cascade)
        each cascade -> user_str, timestamp, user_str, timestamp, ...
    labels_filename : string
        full path of file with labels (each line is for each cascade's label). -1 for unknown labels
    train_cascade_ids_filename : string
        full path of file with int ids corresponding to train cascades, rest are validation ones
    user_max : int
        top most active users to retain over all cascades
    extra_users_len: int (default = None)
        additional X users to include for every missed cascade with no top active users
    lookback_count : int
        number of users to lookback as potential influencers in the cascade
    edge_thr: int
        number of cascades in which u->can influence v for the edge to be added to graph
    cascade_count: int
        number of train_cascades to retain
    Returns
    -------
    u2idx : dict(int, int)
        mapping from user_str to user_index
    idx2u : np.array(str)
        user_str at each user_index
    train_cascades : np.array((None, 2))
        array of (user_index, timestamp)
    test_cascades : np.array((None, 2))
        array of (user_index, timestamp)
    train_labels : np.array(int)
        -1 for unknown labels
    test_labels : np.array(int)
        -1 for unknown labels
    base_graph : PNEANet (snap package)
        u influences v in at least some train cascade according to lookback_count (u to v edge)
        edge attributes (suvmi) and (suvpl) for set of train cascade indices (index in train_cascades array)

    """
    # read raw data files (observed cascades and labels)
    print("reading dataset...")
    raw_observed_cascades = _read_cascades_file(cascades_filename)
    train_cascade_ids = _read_train_cascade_ids_file(train_cascade_ids_filename)
    raw_labels = _read_labels_file(labels_filename)
    test_mask = np.ones(raw_labels.shape, dtype=bool)
    test_mask[train_cascade_ids] = False
    print("indexing dataset...")
    # prune and index
    u2idx, idx2u = _build_user_index(raw_observed_cascades, user_max, extra_users_len)
    indexed_cascades = _convert_cascades_by_index(raw_observed_cascades, u2idx)
    print("filter by train_ids...")
    # filter by train ids
    train_cascades = indexed_cascades[train_cascade_ids]
    test_cascades = indexed_cascades[test_mask]
    train_labels = raw_labels[train_cascade_ids]
    test_labels = raw_labels[test_mask]
    # keep specified num of cascades
    train_cascades = train_cascades[:cascade_count]
    train_labels = train_labels[:cascade_count]
    # remove empty train cascades
    cas_lens = np.array([len(cas) for cas in train_cascades])
    filtered_train_cids = np.where(cas_lens >= min_cas_length)[0]
    train_cascades = train_cascades[filtered_train_cids]
    train_labels = train_labels[filtered_train_cids]
    uniq, uniq_counts = np.unique(train_labels, return_counts=True)
    print('label distribution after discarding small cascades:', uniq, uniq_counts)
    print("creating base_graph...")
    # create base graph from train cascades
    base_graph = _create_base_graph(train_cascades, idx2u, lookback_count, edge_thr)
    return u2idx, idx2u, train_cascades, train_labels, \
        filtered_train_cids, test_cascades, test_labels, base_graph
