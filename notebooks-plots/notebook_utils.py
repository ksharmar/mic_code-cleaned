import numpy as np
import operator
import pandas as pd


def read_cascades_file(cascades_filename):
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
        t = list(map(float, u_t[1::2]))  # float
        cascade = np.vstack([u, t]).transpose()
        cascades.append(cascade)
    f.close()
    return cascades


def indexed_cascades(raw_cascades):
    cascades = []
    for cas in raw_cascades:
        
        u_t = line.strip("\n").split(",")
        u = list(map(int, u_t[0::2]))  # int
        t = list(map(float, u_t[1::2]))  # float
        cascade = np.vstack([u, t]).transpose()
        cascades.append(cascade)
    return cascades


def get_engagement_counts(true_cascades, fake_cascades):
    # distribution of engagement counts (returns user_ids)
    u_t = {}
    for cas in true_cascades:
        for u in cas[:,0]:
            u = int(u)
            if u in u_t: u_t[u] += 1
            else: u_t[u] = 1
    sorted_t = np.array(sorted(u_t.items(), key=operator.itemgetter(1), reverse=True), dtype=np.int32)

    u_f = {}
    for cas in fake_cascades:
        for u in cas[:,0]:
            u = int(u)
            if u in u_f: u_f[u] += 1
            else: u_f[u] = 1
    sorted_f = np.array(sorted(u_f.items(), key=operator.itemgetter(1), reverse=True), dtype=np.int32)
    return u_t, u_f, sorted_t, sorted_f


def eng_count(train_cascades, train_labels):
    true_cascades = np.array(train_cascades)[train_labels==0]
    fake_cascades = np.array(train_cascades)[train_labels == 1]
    print('t/f', len(true_cascades), len(fake_cascades))

    u_t, u_f, sorted_t, sorted_f = get_engagement_counts(true_cascades, fake_cascades)
    print('u_t, u_f, tot', len(u_t), len(u_f), len(u_t) + len(u_f))
    
    u = {**u_t , **u_f}
    sorted_users = np.array(sorted(u.items(), key=operator.itemgetter(1), reverse=True), dtype=np.int32)
    # sorted_users = np.concatenate([sorted_t, sorted_f], axis=0)
    # users with engagements greater than 5
    num_users_with_large_eng = len(sorted_users[sorted_users[:, 1] > 5])
    print('users large eng > 5', num_users_with_large_eng)
    num_users_with_large_eng = len(sorted_users[sorted_users[:, 1] > 10])
    print('users large eng > 10', num_users_with_large_eng)
    
    return u_t, u_f, sorted_users


def get_relative_appearance_in_fake(u_t, u_f, true_infl, fake_infl):
    # inputs contain user ids
    value_true, value_fake = [], []
    for u in true_infl:
        t_count = 0
        f_count = 0 
        i = int(u) # i = str(u)
        if i in u_t:
            t_count = u_t[i]
        if i in u_f:
            f_count = u_f[i]
        tot = t_count + f_count
        if tot == 0:
            continue
        # print("u", u, t_count, f_count, 1.0*t_count/tot, 1.0*f_count/tot)
        value_true.append(100.0 * f_count/tot)
    # print("brk")
    for u in fake_infl:
        t_count = 0
        f_count = 0 
        i = int(u) # i = str(u)
        if i in u_t:
            t_count = u_t[i]
        if i in u_f:
            f_count = u_f[i]
        tot = t_count + f_count
        if tot == 0:
            continue
        # print("u", u, t_count, f_count, 1.0*t_count/tot, 1.0*f_count/tot)
        value_fake.append(100.0 * f_count/tot)
    vt, vf = np.array(value_true), np.array(value_fake)
    ind = np.where(vt == 100)[0]
    # print(ind)
    # vt[ind] = 0
    # print(true_infl[ind])
    return vt, vf
    

