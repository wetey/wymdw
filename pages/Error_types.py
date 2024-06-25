import streamlit as st
import pandas as pd
import numpy as np
import nltk 
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from spacy.lang.en.stop_words import STOP_WORDS
import spacy
from tqdm import tqdm
import string
import collections
import math
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

nltk.download('wordnet')
nltk.download('punkt')
en = spacy.load('en_core_web_sm')
stopwords = en.Defaults.stop_words 
stopwords.update(['film', 'movie', 'nt', 'like','','ve','films'])

def tokenize(document):
  tokens = []
  sentences = sent_tokenize(document)
  for sentence in sentences:
    words = word_tokenize(sentence)

    #make all words lower case
    words = [word.lower() for word in words if word and len(word) >  2]

    #remove all punctuation
    words = [word.translate(str.maketrans('', '', string.punctuation)) for word in words]

    #remove stop words
    words = [word for word in words if word.lower() not in stopwords]
    
    tokens += words
  return tokens

def get_top_50(dataset):

    embeddings = np.array(dataset.embedding.tolist())
    Kmeans_clusterer = KMeans(n_clusters = 1, 
                                random_state = 42)
    clusters = Kmeans_clusterer.fit_predict(embeddings)
    dataset['cluster'] = pd.Series(clusters, index = dataset.index).astype('int')
    dataset['centroid'] = dataset.cluster.apply(lambda x: Kmeans_clusterer.cluster_centers_[x])
    dataset['embedding'] = dataset['embedding'].apply(np.array)
    dataset['distance'] = dataset.apply(lambda x : cosine(x['centroid'], x['embedding']), axis = 1)
    dataset = dataset.groupby('cluster')
    dataset = dataset.apply(lambda x : x.sort_values(by = 'distance'))
    
    return dataset.text.tolist()[:50]

st.set_page_config(page_title="Clustering")
dataset = pd.read_json('data/updated_baseline.json')

st.markdown(
    """
    Now we are going to look at more model specific features of the errors (the false positives and false negatives).

    The false positives refer to examples that were misclassified as positive (the correct label is negative) and the false negatives refer to examples that were misclassified as negative (the correct label is positive).

    The treemap below shows the the most frequent words for the false positives (blue) and the false negatives (red). The size and color of the box is proportional to the log odd ratio. 
        
    """)

dataset = pd.read_json('data/updated_baseline.json')

false_positives = dataset[dataset.predicted == 1.0]
top_50_false_positives = get_top_50(false_positives)

false_negatives = dataset[dataset.predicted == 0.0]
top_50_false_negatives = get_top_50(false_negatives)

top_50_false_positives_documents = []
for document in top_50_false_positives:
  top_50_false_positives_documents.extend(tokenize(document))


top_50_false_negatives_documents = []
for document in top_50_false_negatives:
  top_50_false_negatives_documents.extend(tokenize(document))

unique_tokens = {
                'negatives': collections.defaultdict(int),
                'positives':collections.defaultdict(int)
            }
tokens_in_corpus = collections.defaultdict(int)
for sentence in top_50_false_positives_documents:
    unique_tokens['positives'][sentence] += 1
    tokens_in_corpus[sentence] += 1

for sentence in top_50_false_negatives_documents:
    unique_tokens['negatives'][sentence] += 1
    tokens_in_corpus[sentence] += 1

len_negatives = len(unique_tokens['negatives'])
len_positives = len(unique_tokens['positives'])
len_total = len(tokens_in_corpus)
delta = collections.defaultdict(float)
variance = collections.defaultdict(float)
z_score = collections.defaultdict(float)

for word in set(unique_tokens['negatives']) | set(unique_tokens['positives']):
    
    positive_count = unique_tokens['positives'][word]
    negative_count = unique_tokens['negatives'][word]
    total_count = tokens_in_corpus[word]

    positive_log = math.log10((positive_count + total_count) 
                           / (len_positives + len_total - positive_count - total_count))
    negative_log = math.log10((negative_count + total_count) 
                           / (len_negatives + len_total - negative_count - total_count))
    
    delta[word] = positive_log - negative_log

    positive_fraction = 1 / (positive_count + total_count)
    negative_fraction = 1 / (negative_count + total_count)

    variance[word] = positive_fraction + negative_fraction

    z_score[word] = delta[word] / variance[word]

z_score = sorted(z_score.items(), key = lambda x:x[1], reverse = True)

words = [x[0] for x in z_score[:20]] + [x[0] for x in z_score[-20:]]
logs = [x[1] for x in z_score[:20]] + [x[1] for x in z_score[-20:]]
logs = [x + 5 for x in logs]

top_30_positive_words = [x[0] for x in z_score[:30]]
top_30_positive_logs = [x[1] for x in z_score[:30]]
top_30_negative_words = [x[0] for x in z_score[-30:]]
top_30_negative_logs = [abs(x[1]) for x in z_score[-30:]]

top_30_positive_words = pd.Series(data = top_30_positive_words, name = 'word')
top_30_positive_logs = pd.Series(data = top_30_positive_logs, name = 'log')

top_30_negative_words = pd.Series(data = top_30_negative_words, name = 'word')
top_30_negative_logs = pd.Series(data = top_30_negative_logs, name = 'log')


positive_word_ratio = pd.DataFrame({top_30_positive_words.name: top_30_positive_words, 
                                    top_30_positive_logs.name: top_30_positive_logs})

negative_word_ratio = pd.DataFrame({top_30_negative_words.name: top_30_negative_words, 
                                    top_30_negative_logs.name: top_30_negative_logs})

fig_pos = px.treemap(positive_word_ratio, path=['word'], 
                 values ='log',
                 color_continuous_scale = 'blues',
                 color = 'log',
                 width = 1000,
                 height = 1000)
fig_pos.update_layout(legend= dict(font = dict(size = 14, color = 'black')))
st.plotly_chart(fig_pos)


fig_neg = px.treemap(negative_word_ratio, path=['word'], 
                 values ='log',
                 color_continuous_scale = 'reds',
                 color = 'log',
                 width = 1000,
                 height = 1000)
fig_neg.update_layout(legend = dict(font = dict(size = 14, color = 'black')))
st.plotly_chart(fig_neg)


# words = pd.Series(data = words, name = 'words')
# logs = pd.Series(data = logs, name = 'logs')

# word_ratio = pd.DataFrame({words.name: words, logs.name: logs})

# fig = go.Figure(go.Treemap(
#                  labels = word_ratio.words, 
#                  values = word_ratio.logs,
#                  marker = dict(colorscale = 'rdbu')))
# fig = px.treemap(word_ratio, path=['words'], 
#                  values ='logs',
#                  color_continuous_scale = 'rdbu',
#                  color = 'logs',
#                  width = 1000,
#                  height = 1000)
# st.plotly_chart(fig)
