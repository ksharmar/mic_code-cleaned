"""
# Input data format:
# dataframe object with following columns:
# website    fact    object    sentiment_score
# website = user_id
# object = fact = event name
# sentiment_score = average sentiment score of all user=user_id
#                   tweets on that event

#sample
        website                 fact               object  sentiment_score         
0      15685307          N_Airfrance          N_Airfrance         -0.29600         
1      16513771          N_Airfrance          N_Airfrance         -0.69080         
2      17491506          N_Airfrance          N_Airfrance         -0.29600         
3      17968320          N_Airfrance          N_Airfrance          0.00000         
4       1839718          N_Airfrance          N_Airfrance         -0.29600         
5      19798210          N_Airfrance          N_Airfrance         -0.27320         
6      22258569          N_Airfrance          N_Airfrance         -0.31820         
7      23122005          N_Airfrance          N_Airfrance         -0.29600
"""
import csv
import pdb
import os
import math
import pickle
import time
import argparse
import pandas as pd
import numpy as np
from numpy.linalg import norm
from pprint import pprint
from sklearn.metrics import classification_report
from sklearn import metrics

import warnings
warnings.filterwarnings("error")

class TruthFinder(object):

    def update_fact_confidence(self, df):
        #since we have object == facts, we don't need to have two loops
        for object_ in df["object"].unique():
            indices = df["object"] == object_
            ts = df.loc[indices, "trustworthiness"].values
            sentiment_score = df.loc[indices, "sentiment_score"].values
            df.loc[indices, "fact_confidence"] = np.dot(ts , sentiment_score)/ len(ts)
        return df

    def update_website_trustworthiness(self, df):
        for website in df["website"].unique():
            indices = df["website"] == website
            cs = df.loc[indices, "fact_confidence"]
            sentiment_score = df.loc[indices, "sentiment_score"]
            df.loc[indices, "trustworthiness"] = np.dot(sentiment_score, cs) / len(cs)
        return df

    def iteration(self, df):
        df = self.update_fact_confidence(df)
        df = self.update_website_trustworthiness(df)
        return df

    def stop_condition(self, t1, t2, threshold):
        print("change in trustworthiness of users: {}".format(norm(t2-t1)))
        return norm(t2-t1) < threshold

    def train(self, dataframe, max_iterations=200, threshold=1e-3, initial_trustworthiness=0.9):
        
        dataframe["trustworthiness"] = np.ones(len(dataframe.index)) * initial_trustworthiness
        dataframe["fact_confidence"] = np.zeros(len(dataframe.index))
        scores = {}
        
        for i in range(max_iterations):
            time1 = time.time()
            print("################################## iteration {} ##################################".format(i))
            
            t1 = dataframe.drop_duplicates("website")["trustworthiness"]
            dataframe = self.iteration(dataframe)
            t2 = dataframe.drop_duplicates("website")["trustworthiness"]
            
            if self.stop_condition(t1, t2, threshold):
                return dataframe, scores
            
            scores[i] = self.eval(dataframe)
            time2 = time.time()
            print("iteration {} time = {} seconds".format(i,round(time2-time1,3)))
        return dataframe, scores

    def _metrics_report_to_df(self, ytrue, ypred):
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(ytrue, ypred)
        acc = metrics.accuracy_score(ytrue, ypred)
        classification_report = pd.concat(map(pd.DataFrame, [[acc,acc], fscore, precision, recall, support]), axis=1)
        classification_report.columns = ["accuracy", "f1-score", "precision", "recall", "support"]
        return(classification_report)
    
    def eval(self, dataframe, thresh=0.0):
        groups = dataframe.groupby(by=["fact"])
        y_true, y_pred = [], []
        for name,g in groups:
            print(name,",",g["fact_confidence"].unique())
            score = g["fact_confidence"].unique()[0]
            y_pred.append(0 if score >= thresh else 1)
            y_true.append(0 if name.startswith('N') else 1)
        report = self._metrics_report_to_df(y_true, y_pred)
        # classification_report(y_true, y_pred, target_names=['Not Rumor','Rumor'], output_dict=True)
        pprint(report)
        return report

def main():
    t1 = time.time()
    
    # Create parser
    parser = argparse.ArgumentParser(description='Visualize trajectories')
    # Add arguments
    parser.add_argument('-d', '--data', help='input dataset file',
                        default='preprocess_data/stanford.csv')
    parser.add_argument('-r', '--result-file', help='dump results to this pickle file',
                        default='results.pkl')
    parser.add_argument('-m', '--max-iter', help='maximum iterations for training',
                       default=200, type=int)
    parser.add_argument('-th', '--stop-thresh', help='maximum difference in value\
                        of trustworthiness between two iterations',
                       default=1e-3, type=float)
    parser.add_argument('-t', '--initial-trust', help='initial trustworthiness of \
                        websites', default=0.9, type=float)
    # Parse arguments
    args = parser.parse_args()

    print(args)

    # read data
    dataframe = pd.read_csv(args.data)
    print("Inital state")
    print(dataframe)
    
    finder = TruthFinder()
    dataframe, scores = finder.train(dataframe, max_iterations=args.max_iter,
                                     threshold=args.stop_thresh,
                                     initial_trustworthiness=args.initial_trust)
    
    with open(args.result_file,"wb") as f:
        pickle.dump({'dataframe':dataframe,'scores':scores}, f)
    t2 = time.time()
    print("Program finished in {} seconds".format(round(t2-t1,3)))
    
    
if __name__=="__main__":
    main()
    
    # python baseline_TF.py -d 'kwon/sentiments.txt' -r './tf_kwon_results.pkl'
    # python baseline_TF.py -d 'twitter-ma/twitter-ma_10k.csv' -r './tf_tma_results.pkl'