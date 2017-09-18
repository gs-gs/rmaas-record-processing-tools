from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import nltk
import re


class ESInterface(object):
    
    def __init__(self, url, username, password):
        """ Initializes object, creates ES connection """
        # Create the ES connection
        self.conn = Elasticsearch(url, verify_certs=True, http_auth=(username, password))
        self.index_name = 'digitalrecords-search'
        self.doc_type = 'modelresult'
        # Query to get records that have keywords
        self.query = {
           'query': {
              'exists': {
                 'field': 'keywords'
              }
           },
           '_source': ['id', 'keywords'],
           'size': 1000
        }
        # Get the total number of matching records
        self.get_total()
        
    def get_total(self):
        """ Gets the total number of records that match the query """
        query = self.query
        query['size'] = 1
        search = self.conn.search(body=query, index=self.index_name)
        self.total = search['hits']['total']
        
    def get_records(self, test_run=False):
        print('Total matches: %s' % self.total)
        results = []
        records = scan(self.conn, index=self.index_name, query=self.query, doc_type=self.doc_type)
        if test_run:
            records_num = 50000
        else:
            records_num = self.total
        if test_run:
            print('Total records to fetch: %s (TEST RUN)' % records_num)
        else:
            print('Total records to fetch: %s' % records_num)
        print('\nNow fetching records from ES...')
        for counter, record in enumerate(records, 1):
            if counter > records_num:
                break
            else:
                results += [record['_source']]
            if counter % 10000 == 0:
                print(' - Fetched %s records out of %s' % (counter, records_num))
        print('Finished fetching records from ES\n')
        return results


def tokenize(text):
    # get the english stopwords
    nltk.download('stopwords', quiet=True)
    stopwords = nltk.corpus.stopwords.words('english')
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    nltk.download('punkt', quiet=True)
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.wordpunct_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        # include only those that contains letters
        if re.search('[a-zA-Z]', token):
            # exclude stop words, those shorter than 3 characters, and those that
            # start with non-alphanumeric characters
            if token not in stopwords and len(token) > 2 and token[0].isalnum():
                filtered_tokens.append(token)
    return filtered_tokens
