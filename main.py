#!/usr/bin/env python3

import argparse
import logging as log
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

def main():
    # Goal of program: train multiple classifiers and cross-validate each.
    # Output scores of classifiers on a test set.

    args = parse_args()
    log.basicConfig(level=args.log_level)

    X = pd.read_csv('full-corpus.csv')

    y = X['Sentiment']
    X.drop('Sentiment', axis=1, inplace=True)
    y_classes = list(set(y))
    Y = label_binarize(y, classes=y_classes)
    Y = pd.DataFrame(Y, columns=y_classes)

    X.drop(['TweetId', 'TweetDate'], axis=1, inplace=True)

    lb = LabelBinarizer()
    X_topic = X['Topic']
    X_topic_bin = lb.fit_transform(X_topic)
    X_topic_bin = pd.DataFrame(X_topic_bin, columns=lb.classes_)

    X.drop('Topic', axis=1, inplace=True)
    X = pd.concat([X, X_topic_bin], axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                        test_size=0.2,
                                                        random_state=42)

    for df in X_train, X_test, Y_train, Y_test:
        df.reset_index(drop=True, inplace=True)

    bayes_clf = NBTweetClassifier()

    bayes_clf.fit(X_train, Y_train)
    Y_predict = bayes_clf.predict(X_test)

    #score = accuracy_score(Y_test, Y_predict)
    #print(f"NBClassifier + OvA score: {score}")

if __name__ == '__main__':
    main()
