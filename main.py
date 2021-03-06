#!/usr/bin/env python3

import argparse
import logging as log
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, label_binarize

from NBTweetClassifier import NBTweetClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--debug',
        help="Print debug information",
        action='store_const', dest='log_level', const=log.DEBUG,
        default=log.WARNING
    )
    return parser.parse_args()

def load_dataset(name):
    log.debug(f"Loading {name} dataset")
    if name == 'san-analytics':
        return load_san_analytics_dataset()
    elif name == 'stanford':
        return load_stanford_dataset()

def load_san_analytics_dataset():
    X = pd.read_csv('san-analytics/full-corpus.csv')

    y = X['Sentiment']
    X.drop('Sentiment', axis=1, inplace=True)

    X.drop(['TweetId', 'TweetDate'], axis=1, inplace=True)
    X.drop('Topic', axis=1, inplace=True) # TODO: Make use of this column.

    '''
    lb = LabelBinarizer()
    X_topic = X['Topic']
    X_topic_bin = lb.fit_transform(X_topic)
    X_topic_bin = pd.DataFrame(X_topic_bin, columns=lb.classes_)

    X.drop('Topic', axis=1, inplace=True)
    X = pd.concat([X, X_topic_bin], axis=1)
    '''

    X.rename(columns={'TweetText': 'tweet_text'}, inplace=True)
    y.rename('sentiment', inplace=True)

    return X, y

def load_stanford_dataset():
    X = pd.read_csv('stanford/traindata.csv',
                    header=None,
                    usecols=[0, 5],
                    names=['sentiment', 'tweet_text'],
                    encoding='latin-1')

    y = X['sentiment']
    X.drop('sentiment', axis=1, inplace=True)

    return X, y

def main():
    # Goal of program: train multiple classifiers and cross-validate each.
    # Output scores of classifiers on a test set.

    args = parse_args()
    log.basicConfig(level=args.log_level)

    for ds_name in (
                    'san-analytics',
                    #'stanford'
                    ):
        X, y = load_dataset(ds_name)

        log.debug("Splitting train and test data")
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=42
                                                            )
        for df in X_train, X_test, y_train, y_test:
            df.reset_index(drop=True, inplace=True)

        bayes_clf = NBTweetClassifier()

        bayes_clf.fit(X_train, y_train)
        y_predict = bayes_clf.predict(X_test)

        score = accuracy_score(y_test, y_predict)
        print(f"{ds_name} | NBTweetClassifier score: {score}")

        good_path = f'{ds_name}/good_predict.csv'
        log.debug(f"Dumping good predictions to {good_path}")
        good_predict_index = np.argwhere(y_predict == y_test).ravel()
        good_predict = pd.concat([X_test.iloc[good_predict_index],
                                  y_predict[good_predict_index]
                                  ], axis=1)
        good_predict.to_csv(good_path, encoding='utf-8', index=False)

        bad_path = f'{ds_name}/bad_predict.csv'
        log.debug(f"Dumping bad predictions to {bad_path}")
        bad_predict_index = np.argwhere(y_predict != y_test).ravel()
        bad_predict = pd.concat([X_test.iloc[bad_predict_index],
                                 y_test[bad_predict_index].rename('sentiment_expected'),
                                 y_predict[bad_predict_index].rename('sentiment_actual')
                                 ], axis=1)
        bad_predict.to_csv(bad_path, encoding='utf-8', index=False)

if __name__ == '__main__':
    main()
