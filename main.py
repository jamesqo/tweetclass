#!/usr/bin/env python3

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def main():
    # Goal of program: train multiple classifiers and cross-validate each.
    # Output scores of classifiers on a test set.
    # Multiclass prediction will be handled using OvA.

    X = pd.read_csv('full-corpus.csv')
    y = X['Sentiment']
    X.drop('Sentiment', axis=1, inplace=True)

    X.drop(['TweetId', 'TweetDate'], axis=1, inplace=True)

    lb = LabelBinarizer()
    X_topic = X['Topic']
    X_topic_bin = lb.fit_transform(X_topic)
    X_topic_bin = pd.DataFrame(X_topic_bin, columns=lb.classes_)

    X.drop('Topic', axis=1, inplace=True)
    X = pd.concat([X, X_topic_bin], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    print(X_train.head(n=10))
    print(y_train.head(n=10))

    #bayes_clf = NBClassifier()
    #bayes_clf.

if __name__ == '__main__':
    main()
