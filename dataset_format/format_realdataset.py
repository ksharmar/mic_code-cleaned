import os, sys
import time, datetime
import numpy as np
from email.utils import parsedate_tz, mktime_tz


def getCascadeSequences_and_SeqLabel():
    
    #-----------------------------
    
    # savepath="./data/kwon/"
    # path="kwon_dataset/"

    # savepath="pheme-rnr-dataset/"
    # path="pheme-rnr-dataset/"

    savepath="twitter-ma/"
    path="twitter-ma/"

    # savepath="twitter-qian/"
    # path="twitter-qian/"

    #-----------------------------
    
    # links = "links.txt" # uid_A (follows) uid_B
    cascades = "Tweets/" # uid, tid, content, time (each file is a diffusion cascade)
    # user_feats = "sub_user_info_share.txt" # uid, nbfollowers, nbfollowees, tweets 
    # user_feats = "user_data.csv"
    
    #----------------------------- GET (uid, time)

    diff_cascades = {}
    labels = []

    for tweetsfile in os.listdir(path+cascades):
        f = open(path+cascades+tweetsfile, "r")
        # print tweetsfile
        print tweetsfile
        ulist, tslist = list(), list()
        for line in f.readlines():
            uid, timestamp = line.split("\t")[0], int(line.split("\t")[-1])
            
#             # twitter ma and kwon format:            
#             uid, timestamp = line.split("\t")[0], time.mktime(datetime.datetime.strptime(
#                 line.split("\t")[-1].rstrip(), '%Y-%m-%d %H:%M:%S').timetuple())
            
#             # pheme format
#             uid, timestamp = line.split("\t")[0], mktime_tz(parsedate_tz(line.split("\t")[-1].rstrip()))

            ulist.append(uid)
            tslist.append(timestamp)
        tslist, ulist = zip(*sorted(zip(tslist, ulist)))
        rel_tslist = np.array(tslist) - np.array(tslist[0])
        if len(ulist) <= 1: continue
        diff_cascades[tweetsfile] = (ulist, rel_tslist)
        # print diff_cascades
        # break
    # print diff_cascades
    
    #----------------------------- Write (uid, time, ...) and labels

    w = open(savepath+"/train_cascades.txt", "w")
    for tweetsfile, tweets in diff_cascades.items():
        ulist = tweets[0]
        tlist = tweets[1]
        label = 1 if tweetsfile.strip().split("_")[0] == 'R' else 0

        wstring = ""
        for i in range(len(ulist)):
            wstring+= str(ulist[i])+"-"+str(tlist[i])+" "
        # w.write(" ".join(tweets, rel_tslist) + "\n")
        # print wstring.strip()
        if wstring.strip() == "": continue
        labels.append(label)
        w.write(wstring.strip() + "\n")

    np.savetxt(savepath + "train_labels.txt", np.array(labels), fmt="%d")


def get_cascade_filenames_list():
    #-----------------------------
    
    # savepath="./data/kwon/"
    path="kwon/

    # savepath="pheme-rnr-dataset/"
    # path="pheme-rnr-dataset/"

    # savepath="twitter-ma/"
    # path="twitter-ma/"

    # savepath="twitter-qian/"
    # path="twitter-qian/"

    #-----------------------------
    
    # links = "links.txt" # uid_A (follows) uid_B
    cascades = "Tweets/" # uid, tid, content, time (each file is a diffusion cascade)
    # user_feats = "sub_user_info_share.txt" # uid, nbfollowers, nbfollowees, tweets 
    # user_feats = "user_data.csv"
    
    #----------------------------- GET (uid, time)

    diff_cascades = {}
    labels = []

    for tweetsfile in os.listdir(path+cascades):
        f = open(path+cascades+tweetsfile, "r")
        # print tweetsfile
        print tweetsfile
        
    #----------------------------- Write (uid, time, ...) and labels

    w = open(savepath+"/train_cascades.txt", "w")
    for tweetsfile, tweets in diff_cascades.items():
        ulist = tweets[0]
        tlist = tweets[1]
        label = 1 if tweetsfile.strip().split("_")[0] == 'R' else 0

        wstring = ""
        for i in range(len(ulist)):
            wstring+= str(ulist[i])+"-"+str(tlist[i])+" "
        # w.write(" ".join(tweets, rel_tslist) + "\n")
        # print wstring.strip()
        if wstring.strip() == "": continue
        labels.append(label)
        w.write(wstring.strip() + "\n")

    np.savetxt(savepath + "train_labels.txt", np.array(labels), fmt="%d")
    
    
    
def fixFormat_dash_to_comma_CascadesFile():
    f = open('train_cascades.txt')
    w = open('cascades.txt', 'w')
    for line in f.readlines():
        ut_pairs = line.strip().split(" ")
        s = ''
        for ut in ut_pairs:
            u, t = ut.split("-")
            s += u + ',' + t + ','
        s = s.strip(',')
        w.write(s + '\n')