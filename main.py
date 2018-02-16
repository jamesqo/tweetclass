#!/usr/bin/env python3

import pandas as pd

from sklearn.model_selection import train_test_split

def main():
    # Goal of program: train multiple classifiers and cross-validate each.
    # Output scores of classifiers on a test set.

    X = pd.read_csv('full-corpus.csv')
    y = X['Sentiment']
    X.drop('Sentiment', axis=1, inplace=True)

    #print(X.head(n=10))
    #print(y.head(n=10))

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    #print(X_train.head(n=10))
    #print(y_train.head(n=10))

if __name__ == '__main__':
    main()
