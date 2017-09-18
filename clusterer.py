#!/usr/bin/env python

from sklearn.feature_extraction.text import TfidfVectorizer
from helpers import ESInterface, tokenize
from sklearn.cluster import KMeans
import pandas as pd
import pickle
import os


class Clusterer(object):
    
    def __init__(self, es_credentials, nclusters=10, test_run=False):
        self.nclusters = nclusters
        self.test_run = test_run
        url, username, password = es_credentials
        self.elasticsearch = ESInterface(url, username, password)
        
    def build_dataframe(self):
        records = self.elasticsearch.get_records(test_run=self.test_run)
        df = pd.DataFrame(records)
        df.fillna('', inplace=True)
        self.df = df
    
    def build_tfidf_matrix(self, content_column='keywords'):
        # convert the content column into string values, to be safe
        self.df[content_column] = self.df[content_column].astype(str)
        # prepare the feature extraction vectorizer 
        self.vectorizer = TfidfVectorizer(
            max_features=200000,
            stop_words='english',
            use_idf=True, 
            tokenizer=tokenize)
            
        # extract the features and get the feature matrix
        self.feature_matrix = self.vectorizer.fit_transform(self.df[content_column])
        self.feature_names = self.vectorizer.get_feature_names()
        
    def perform_kmeans_clustering(self):
        # perform the k-means clustering
        self.km = KMeans(n_clusters=self.nclusters)
        self.km.fit(self.feature_matrix)
        clusters = self.km.labels_.tolist()
        docs = { 'id': self.df['id'], 'cluster': clusters }
        clusters_df = pd.DataFrame(docs, columns = ['id', 'cluster'])
        return clusters_df
        
    def get_keywords_per_cluster(self, max_num=20):
        order_centroids = self.km.cluster_centers_.argsort()[:, ::-1]
        clusters = range(self.nclusters)
        top_terms = {k: [] for k in clusters}
        for i in clusters:
            cluster_top_terms = [self.feature_names[x] for x in order_centroids[i, :max_num]]
            top_terms[i] = cluster_top_terms
        return top_terms
        
    def describe_clusters(self, clusters, cluster_terms):
        cdf = clusters.groupby('cluster').agg('count')
        cdf.rename(columns={'keywords': 'count'}, inplace=True)
        cdf['keywords'] = [cluster_terms[x] for x in cdf.index]
        return cdf


def execute(url, username, password, nclusters=10, test_run=False):
    print('\nConnecting to ElasticSearch...')
    cl = Clusterer(
        [url, username, password],
        nclusters=nclusters,
        test_run=test_run)
    print('Building a dataframe of document IDs and keywords...')
    cl.build_dataframe()
    print('Generating the TF-IDF matrix...')
    cl.build_tfidf_matrix()
    print('Performs K-Means clustering...')
    clusters_df = cl.perform_kmeans_clustering()
    top_terms_df = cl.get_keywords_per_cluster()
    # combine the records and clusters dataframes
    cl.df.set_index('id', inplace=True)
    clusters_df.set_index('id', inplace=True)
    combined_df = cl.df.join(clusters_df, how='outer')
    summary = cl.describe_clusters(combined_df, top_terms_df)
    return (cl.vectorizer, cl.feature_matrix, combined_df, top_terms_df, summary)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Performs document clustering')
    parser.add_argument('-U', '--url', type=str, 
        help='ElasticSearch URL', required=True)
    parser.add_argument('-u', '--username', type=str, 
        help='ElasticSearch username', required=True)
    parser.add_argument('-p', '--password', type=str, 
        help='ElasticSearch password', required=True)
    parser.add_argument('-o', '--output', type=str, 
        help='Path to output file', required=True)
    parser.add_argument('--test-run', action='store_true',
        help='Test run the clustering with 50,000 records')
    args = parser.parse_args()
    vectorizer, features, clusters_df, top_terms_df, summary= execute(
        args.url, args.username, args.password, test_run=args.test_run)
    print('Writing the output files...')
    results = {
        'vectorizer': vectorizer,
        'features': features,
        'clusters': clusters_df,
        'terms': top_terms_df
    }
    pickle.dump(results, open(args.output, 'wb'))
    print('\nClustering finished!\n')
    print('Results:\n', summary, '\n')