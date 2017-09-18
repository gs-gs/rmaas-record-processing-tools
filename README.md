# record-processing-tools

Reference implementation of tools for processing formal records.

This repository contains two modules for automated keywords-based clustering and classification of records. These are provided as annotated notebooks that demonstrate and explain the techniques used, and also as simple programs that can be used from the command line.

Please note that the purpose of this software is to demonstrate that it is possible to build a smart system that accesses a records management platform through APIs, and provides value-adding "smart services" using machine learning and artificial intelligence tools.

It is not a production-ready system.


## Using the clustering module

The `clusterer.py` module is an executable script that performs document clustering. The records are fetched from ElasticSearch (index name: digitalrecords-search). The script assumes that keywords have been extracted from the documents and stored unders keywords field. The script only works on the documents that have keywords.

To execute a clustering run, you need to pass ES credentials and output file path to the script. An example command is given below:

`./clusterer.py --url [https://example.esclusterhost.com:9243/] --username [username] --password [password] --output [path to clustering output] --test-run`<br>

The command above has the `--test-run` flag. This will do clustering of the first 50,000 records fetched from ES. Just remove that flag to perform clustering on all the records that contain keywords.

## Using the classifier module
The `classifier.py` module is an executable script that performs document classification. It can be used in two ways -- to build the classifier model or to classify documents using a saved model. Building a classifier model requires the output of the clustering run and path to the classifier model to be saved.

Example command for building a model:

`./classifier.py build_model --clusters ~/Desktop/clusters.pkl --classifier ~/Desktop/classifier.pkl`<br>

To use the generated model for classifying records/documents, the records to be classified has to be written in a CSV file with two columns -- `id` and `keywords`. Sample CSV:

```
id,keywords
records.recordname.cfe6b80e-2b81-4114-9e10-259d6c0c60df,"Cleantech, Addition, cleantech, Association, Archived, Winners, Cluster, highlight, Category:, Categories"
records.recordname.fcfa325c-76b8-4c59-aec0-b1e63f79d559,"Addition, Multi-Asset, Renewal, Archived, Switzerland, program, launches, Categories"
```

Example command to do the classification using a saved model:

`./classifier.py classify --keywords ~/Desktop/keywords.csv --classifier ~/Desktop/classifier.pkl`<br>

#### Sample complete workflow from clustering to classification

Generate the clusters:

`./clusterer.py --url https://b8cecd38087b50c322e84733eb56a855.ap-southeast-2.aws.found.io:9243/ --username <username here> --password <password here> --output ~/Desktop/clusters.pkl --test-run`<br>

Build the classifier model:

`./classifier.py build_model --clusters ~/Desktop/clusters.pkl --classifier ~/Desktop/classifier.pkl`<br>

Classify new records written in `keywords.csv`:

`./classifier.py classify --keywords ~/Desktop/keywords.csv --classifier ~/Desktop/classifier.pkl`<br>

## Running the notebooks
The notebooks contain the proof-of-concept implementations of the clustering and classification algorithms. Follow the following steps to run the notebooks

### Install the required packages
`pip install -r requirements_prod.txt`<br>
`pip install -r requirements_dev.txt`

### Download sample data
`cd notebooks`<br>
`wget -O data/data_contents.zip https://www.dropbox.com/s/p3q7vyhz3e981kr/data_contents.zip?dl=1`<br>
`cd data; unzip -o data_contents.zip; rm data_contents.zip; cd ..`

### Run Jupyter notebook
`jupyter notebook`<br>

Then open your browser to `http://localhost:8888`. <br>
Open a notebook, click on `Cell > Run all` in the top menu.
