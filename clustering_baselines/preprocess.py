import pdb
import os
import math
import time
import argparse
import numpy as np
import pandas as pd

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import open
analyser = SentimentIntensityAnalyzer()

def sentiment_score(sentence):
    score = analyser.polarity_scores(sentence)
    return score['compound']

def preprocess_harvard(datadir, output, num_users=1000):
    data = np.array([]).reshape(-1,4)
    popular_users = pd.DataFrame(columns=['website','tweet_id'])
    for event_file in os.listdir(datadir):
        path = os.path.join(os.getcwd(), datadir, event_file)
        fact_name = event_file.rsplit('.',1)[0]
        with open(path,"r") as f:
            lines = f.readlines()
            line_split = [line.split('	') for line in lines]

            websites = np.array([line[0] for line in line_split]).reshape(-1,1)
            tweet_ids = np.array([line[1] for line in line_split]).reshape(-1,1)
            tweet_score = np.array([sentiment_score(line[2]) for line in line_split]).reshape(-1,1)

            user_df = pd.DataFrame(np.concatenate((websites, tweet_ids, tweet_score), axis=1),
                                   columns=['website','tweet_id','sentiment_score'])
            user_df['sentiment_score'] = user_df['sentiment_score'].astype(float)
            popular_users = popular_users.append(user_df[['website','tweet_id']])
            user_df = user_df.groupby(by=['website'])['sentiment_score'].mean()

            websites = user_df.index.values.reshape(-1,1)
            tweet_score = user_df.values.reshape(-1,1)
            facts = np.array([fact_name for x in range(len(websites))]).reshape(-1,1)
            objects = np.array([fact_name for x in range(len(websites))]).reshape(-1,1)

        #website=users, fact=events, object=events
        cur_data = np.concatenate((websites, facts, objects, tweet_score), axis=1)
        data = np.concatenate((data, cur_data))

    df = pd.DataFrame(data, columns=["website", "fact", "object","sentiment_score"])
    num_users = len(df.index) if num_users==0 else num_users
    popular_user_indices = popular_users.groupby(by=['website']).size().nlargest(n=num_users).index
    df = df.loc[df['website'].isin(popular_user_indices)]
    df.to_csv(output, index=False)



def main():
    t1 = time.time()
    
    # Create parser
    parser = argparse.ArgumentParser(description='Visualize trajectories')
    # Add arguments
    parser.add_argument('-d', '--datadir', help='input data path',
                        default='harvard')
    parser.add_argument('-o', '--output', help='output dataframe file path',
                        default='harvard.csv')
    parser.add_argument('-p', '--popular-users', help='number of popular user\
                        to consider in dataset', type=int, default=0)
    
    # Parse arguments
    args = parser.parse_args()

    preprocess_harvard(args.datadir, args.output, num_users=args.popular_users)
    
    t2 = time.time()
    print("Program finished in {} seconds".format(round(t2-t1,3)))

if __name__=="__main__":
    main()
