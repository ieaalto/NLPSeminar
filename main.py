import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from gensim import models
from collections import Counter
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.cluster import hierarchy
from sklearn.preprocessing import normalize
from settings import *


def load_articles():
    with open(ARTICLES_FILENAME, 'r', encoding='utf8') as f:
        # The original articles JSON file is malformed, hence the json package does not work.
        articles = ast.literal_eval(f.read())

        return {art['url']: art['content'] for art in articles}


def load_annotations():
    with open(ANNOTATIONS_FILENAME, 'r') as f:
        lines = f.readlines()
        n = int(lines[0])
        urls = [line.strip() for line in lines[1:n+1]]
        matrix_lines = [parse_matrix_line(line, urls) for line in lines[n+1:]]

        data = {urls[i]: matrix_lines[i] for i in range(len(urls))}
        return pd.DataFrame.from_dict(data, orient='index')


def parse_matrix_line(line, urls):
    return {urls[i]: line[i] for i in range(len(urls))}


# Converts the similarity matrix to class labels
def get_ground_truth_clusters(annotations):
    classes = dict()

    # class -1 denotes a document not to be included in the evaluation set
    cluster = 0
    k = 0
    for i in annotations.index:
        if i not in classes:
            cluster += 1
            classes[i] = cluster
            for j in annotations.index[k:]:
                relation = annotations.loc[i, j]
                if relation == '+':
                    classes[j] = cluster
                elif relation == '~':
                    classes[i] = -1
                    classes[j] = -1
                elif relation in ['x', 's']:
                    classes[j] = -1
        k += 1
    return classes


def evaluate_clustering():
    pass


def combine_content_with_annotations(articles, annotations):
    docs = annotations.copy()
    docs['content'] = pd.Series([articles[url] if url in articles else "_NOT FOUND_" for url in annotations['url']], index=annotations.index)
    docs['group'] = docs['group'].where(docs['content'] != "_NOT FOUND_", -1)
    return docs[docs['group'] != -1]


# Load and combine articles and class labels. Articles are kept in a separate file due to formatting issues.
# Returns the data in a single data frame.
def load_data():
    annotations = pd.DataFrame.from_csv(PARSED_GROUPINGS_FILENAME)
    articles = load_articles()
    return combine_content_with_annotations(articles, annotations).reset_index()


def load_w2v_model(fname):
    return models.Word2Vec.load(fname)


def get_w2v_tranform(model, tfidf=False, max_features=99999):
    def transform(docs):
        if tfidf:
            vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=max_features)
        else:
            vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english', max_features=max_features)
        weights = vectorizer.fit_transform(docs['content'])
        features = []
        names = vectorizer.get_feature_names()

        for i, doc in docs['content'].iteritems():
            row = weights.getrow(i)
            word_counts = {names[j]: row[0, j] for j in row.nonzero()[1]}

            features.append(np.sum([count * model[word] for word, count in word_counts.items() if word in model.vocab], axis=0))

        return np.array(features)

    return transform


def get_combined_transform(model, alpha=0.82, tfidf=False):
    def transform (docs):
        vects = alpha*normalize(get_w2v_tranform(model, tfidf=tfidf)(docs), axis=1, norm='l1')
        tfidfs = (1-alpha)*tfidf_transform(docs)
        return np.concatenate([tfidfs, vects], axis=1)

    return transform


def tune_alpha(data, model, start=0.05, end=0.5, step=0.05):
    maxs = {}
    for a in np.arange(start, end, step):
        vscores, clusterings = get_vmeasure_curve_and_clusterings(data, get_combined_transform(model, a))
        maxs[a] = max(vscores)
    return maxs


def tfidf_transform(docs, max_features=7500):
    vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=max_features)
    return vectorizer.fit_transform(docs['content']).todense()


def bow_transform(docs):
    vectorizer = CountVectorizer(strip_accents='unicode', stop_words='english', max_features=7500)
    return vectorizer.fit_transform(docs['content']).todense()


# Return the clusterings for the values of t specified in t_range. transform is a function that converts data to the
# format given to the clustering algorithm
def group_documents(docs, transform, t_range):
    data = transform(docs)
    linkage = hierarchy.linkage(data, metric='cosine', method='complete')

    if type(t_range) in (range, np.ndarray, list, tuple):
        groupings = []
        for t in t_range:
            groupings.append(get_clustering_by_t(docs, linkage, t))
        return groupings
    else:
        return get_clustering_by_t(docs, linkage, t_range)


def get_clustering_by_t(docs, linkage, t):
    labels = hierarchy.fcluster(linkage, t, criterion='distance')

    docs_with_clusters = docs.copy()
    docs_with_clusters['cluster'] = labels
    return docs_with_clusters


def group_documents_with_w2v(docs, model, t):
    return group_documents(docs, get_w2v_tranform(model), t)


def evaluate_clustering(data):
    v = metrics.v_measure_score(data['group'], data['cluster'])
    naive = data.copy()
    naive['cluster'] = np.arange(0, naive.shape[0])
    v_naive = metrics.v_measure_score(data['group'], naive['cluster'])
    return (v - v_naive) / (1 - v_naive)


def get_vmeasure_curve_and_clusterings(data, transform, start=0.0001, end=1.0, step=0.001):
    t_range= np.arange(start, end, step)
    clusterings = group_documents(data, transform, t_range)
    vscores = pd.Series([evaluate_clustering(clustering) for clustering in clusterings], index=t_range)
    return vscores, clusterings


def v_measure_figure():
    plt.figure()
    plt.xlabel(r'$\theta$')
    plt.ylabel('Naive-adjusted V-score')
    plt.ylim((0.0, 0.4))


def load_google_news_vectors(filename=GOOGLE_NEWS_VECTS_FILENAME):
    return models.KeyedVectors.load_word2vec_format(filename, binary=True)
