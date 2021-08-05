"""
Input data format:
- train_cascades.txt: each line is a cascade (uid-timestamp uid2-timestamp2 ... ->time-ordered)
- train_labels.txt: corresponding labels

Hyperparameters:
- lag = 24 hours
- cutoff = 10K hours

Output data format:
- seiz_results.txt: (\t separated pandas table - each line corresponds to one line in the input file)  
- 0 = NR (T) and 1 = R (F)
----------------------------
scores	predicted_labels
0.03	1
1.20    0
----------------------------
"""

import numpy as np
# import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from scipy.integrate import odeint
from lmfit import minimize, Parameters, Parameter, report_fit
from scipy.integrate import odeint
import os
import math
import time
import argparse
import pandas as pd
from pprint import pprint
from sklearn import metrics
# np.random.seed(0)

class SEIZ(object):
    
    def _f(self, y, t, paras):
        """ ODEs"""
        S = y[0]
        E = y[1]
        I = y[2]
        Z = y[3]
        N = S + E + I + Z
        beta = paras['beta'].value
        b = paras['b'].value
        l = paras['l'].value
        p = paras['p'].value
        rho = paras['rho'].value
        eps = paras['eps'].value
        f0 = -beta * S * I / N - b * S * Z / N
        f1 = (1-p) * beta * S * I / N + (1-l) * b * S * Z / N - rho * E * I / N - eps * E
        f2 = p * beta * S * I / N + rho * E * I / N + eps * E
        f3 = l * b * S * Z / N
        return [f0, f1, f2, f3]

    def _g(self, t, x0, paras):
        """ Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0 """
        x = odeint(self._f, x0, t, args=(paras,))
        return x

    def _residual(self, paras, t, data):
        """ compute the residual between actual data and fitted data """
        x0 = paras['s'].value, paras['e'].value, paras['i'].value, paras['z'].value
        model = self._g(t, x0, paras)
        # you only have data for one of your variables
        x2_model = model[:, 2]
        return (x2_model - data).ravel()

    def compute_rsi(self, val_measured):
        # # initial conditions
        s0 = len(val_measured) # np.random.uniform(0, 100) # len(val_measured)
        e0 = 0 # np.random.uniform(0, 100)
        i0 = 0 # np.random.uniform(0, 100)
        z0 = 0 # np.random.uniform(0, 100)
        y0 = [s0, e0, i0, z0]
        t_measured = np.linspace(0, len(val_measured), len(val_measured))
        # plt.figure()
        # plt.scatter(t_measured, val_measured, marker='o', color='b', label='measured data', s=75)

        # set parameters including bounds; you can also fix parameters (use vary=False)
        params = Parameters()
        params.add('s', value=s0, min=0.0001)
        params.add('e', value=e0, min=0.0001)
        params.add('i', value=i0, min=0.0001)
        params.add('z', value=z0, min=0.0001)
        params.add('beta', value=np.random.rand(), min=0.0001)
        params.add('b', value=np.random.rand(), min=0.0001)
        params.add('l', value=np.random.rand(), min=0.0001, max=1.0)
        params.add('p', value=np.random.rand(), min=0.0001, max=1.0)
        params.add('rho', value=np.random.rand(), min=0.0001)
        params.add('eps', value=np.random.rand(), min=0.0001)

        # fit model
        # print len(t_measured), len(val_measured)
        result = minimize(self._residual, params, args=(t_measured, val_measured), method='leastsq')
        data_fitted = self._g(np.linspace(0., len(val_measured), len(val_measured)), y0, result.params)
        # print result.params
        # report_fit(result)
        del val_measured
        del data_fitted
        del t_measured
        RSI = ((1-result.params['p'].value) * result.params['beta'].value + (1- result.params['l'].value) * result.params['b'].value) / (result.params['rho'].value + result.params['eps'].value)
        return RSI
    
    def get_i_measured_activations(self, activations, lag, cutoff):
        hour = lag; i_in_period = 0; i_cumulative = []
        # print(len(activations))
        for active in activations:
            # print active
            # print active
            uname, timestamp = active.split("-")
            timestamp = float(timestamp)/3600
            # print timestamp
            if timestamp > cutoff and len(i_cumulative) >= 30: 
                print "breaking"
                break
            if timestamp > hour:
                counter = timestamp / lag - len(i_cumulative) 
                # if counter > 100: counter=100
                for c in range(int(counter)):
                    # print "l", len(i_cumulative)
                    i_cumulative.append(i_in_period)
                    hour += lag
                print "adding", counter
            i_in_period += 1
            # if len(i_cumulative) > 1000: break
        # print len(i_cumulative)
        # print len(i_cumulative)
        return i_cumulative
        
    def process_file(self, filename, lag, cutoff):
        """read all cascade from training or testing files. """
        rsi_list = []
        lines = open(filename).readlines()
        for linenum in range(0, len(lines)):
            line = lines[linenum]
            activations = line.strip().split(" ")
            print "-----------------------------", len(activations)
            i_measured = self.get_i_measured_activations(activations, lag, cutoff)
            # fit and compute rsi using i_measured
            if len(i_measured) < 10: 
                print line
                print len(activations)
                sys.exit()
                rsi = 0.5 # number of parameters in the model
                print "bad observation"
            else: rsi = self.compute_rsi(i_measured) 
            rsi_list.append(rsi)
            print(str(linenum)+":"+str(rsi))
#             if len(rsi_list) == 2:
#                 break
        return rsi_list
        
def _metrics_report_to_df(ytrue, ypred):
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(ytrue, ypred)
    acc = metrics.accuracy_score(ytrue, ypred)
    classification_report = pd.concat(map(pd.DataFrame, [[acc,acc], fscore, precision, recall, support]), axis=1)
    classification_report.columns = ["accuracy", "f1-score", "precision", "recall", "support"]
    return(classification_report)

def main():
    t1 = time.time()
    
    # Create parser
    parser = argparse.ArgumentParser(description='SEIZ')
    # Add arguments
    parser.add_argument('-d', '--datapath', help='cascades file', default='/home///twitter-ma/train_cascades.txt')
    parser.add_argument('-label', '--labelpath', help='labels file', default='/home///twitter-ma/train_labels.txt')
    parser.add_argument('-s', '--savepath', help='results file', default='/home///twitter-ma/seiz_results2.txt')
    parser.add_argument('-l', '--lag', help='', default=24, type=float)
    parser.add_argument('-c', '--cutoff', help='cutoff', default=10000, type=int)
    # Parse arguments
    args = parser.parse_args()
    
    print(args.datapath)
    
    model = SEIZ()
    scores = np.array(model.process_file(args.datapath, args.lag, args.cutoff))
    avg_score = np.mean(scores)
    labels = 1-(scores >= avg_score)*1 # convention: true/NR=0, false/R=1 # high rsi = true news
    ground = np.loadtxt(args.labelpath)# [0:2]
    # assert(len(labels) == len(ground))
    # print scores, predicted labels, labels
    df = pd.DataFrame({'scores': scores, 'predicted_labels': labels, 'ground_labels':ground}, columns=['scores', 'predicted_labels', 'ground_labels'])
    df.to_csv(args.savepath, sep='\t', index=False)
    
    # df = pd.read_csv(args.savepath, sep='\t')
    ground = np.loadtxt(args.labelpath)
    scores = df['scores']
    th = np.median(scores)
    print th
    labels = 1-(scores >= th)*1
    report =_metrics_report_to_df(ground, labels)
    pprint(report)
    t2 = time.time()
    print("Program finished in {} seconds".format(round(t2-t1,3)))
    
    
if __name__=="__main__":
    main()