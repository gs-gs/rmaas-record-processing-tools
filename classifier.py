#!/usr/bin/env python

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import pandas as pd
import pickle
import json


class Classifier(object):
    
    def __init__(self):
        self.classifier = None
        
    def load_training_data(self, clustering_output):
        clustering_output= pickle.load(open(clustering_output, 'rb'))
        self.clusters = clustering_output['clusters']
        self.vectorizer = clustering_output['vectorizer']
        self.feature_matrix = clustering_output['features']
        self.cluster_terms = clustering_output['terms']
        self.feature_names = self.vectorizer.get_feature_names()
        
    def describe_clusters(self):
        cdf = self.clusters.groupby('cluster').agg('count')
        cdf.rename(columns={'keywords': 'count'}, inplace=True)
        cdf['keywords'] = [self.cluster_terms[x] for x in cdf.index]
        return cdf
        
    def build_model(self, outfile, algorithm='svm'):
        # build the classifier model
        classifiers = {
            'naive_bayes': MultinomialNB(),
            'svm': SGDClassifier(loss='hinge', penalty='l2', 
                alpha=1e-3, random_state=42, max_iter=5, tol=None)
        }
        self.classifier = classifiers[algorithm]
        targets = self.clusters['cluster'].values
        self.classifier.fit(self.feature_matrix, targets)
        # save the classifier model into a picke file
        joblib.dump((self.vectorizer, self.classifier), outfile)
        
    def load_model(self, pickle_file):
        # load model from saved pickle file
        self.vectorizer, self.classifier = joblib.load(pickle_file)
        
    def classify(self, texts):
        if self.classifier:
            counts = self.vectorizer.transform(texts)
            predictions = self.classifier.predict(counts)
            return [int(x) for x in predictions]
        else:
            print('A classifier model has not been built or loaded')
            
    def cross_validate(self):
        kf = KFold(n_splits=3)
        scores = []
        pipeline = Pipeline([
            ('vectorizer',  self.vectorizer),
            ('classifier',  self.classifier)
        ])
        for train_indices, test_indices in kf.split(self.clusters):
            train_text = self.clusters.iloc[train_indices]['keywords'].values
            train_y = self.clusters.iloc[train_indices]['cluster'].values
            test_text = self.clusters.iloc[test_indices]['keywords'].values
            test_y = self.clusters.iloc[test_indices]['cluster'].values
            
            pipeline.fit(train_text, train_y)
            predictions = pipeline.predict(test_text)
            score = accuracy_score(test_y, predictions)
            scores.append(score)
            
        print('Total records classified:', len(self.clusters))
        print('Accuracy score:', sum(scores)/len(scores))


def execute(command, clusters=None, keywords=None, classifier=None):
    cl = Classifier()
    if command == 'build_model':
        cl.load_training_data(clusters)
        cl.build_model(classifier)
    elif command == 'classify':
        keywords_df = pd.read_csv(keywords)
        cl.load_model(classifier)
        keywords = keywords_df['keywords'].values
        predictions = cl.classify(keywords)
        results = dict(zip(keywords_df['id'].values, predictions))
        print(json.dumps(results))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Performs document classification')
    parser.add_argument('command', type=str, default='',
        help='The command to execute: [build_model, classify]')
    parser.add_argument('--clusters', type=str, default='',
        help='The path to the clustering output')
    parser.add_argument('--classifier', type=str, default='', 
        help='The path to the saved classifier model')
    parser.add_argument('--keywords', type=str, default='',
        help='A list of keywords of keyword lists in JSON format')
    args = parser.parse_args()
    if args.command == 'build_model':
        if not args.clusters:
            print('\nThe path to the clustering output is required\n')
        if not args.classifier:
            print('\nThe path to the saved classifier model is required\n')
        if args.clusters and args.classifier:
            execute('build_model', clusters=args.clusters, classifier=args.classifier)
            print('\nBuilding classifier model complete. The model has been saved to file.\n')
    elif args.command == 'classify':
        if not args.keywords:
            print('\nThe list of keyword lists in JSON format is required\n')
        if not args.classifier:
            print('\nThe path to the saved classifier model is required\n')
        if args.keywords and args.classifier:
            execute('classify', keywords=args.keywords, classifier=args.classifier)
    elif args.command == '':
        print('\nA specific command is required\n')