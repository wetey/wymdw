import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.manifold import TSNE 
from sklearn.decomposition import PCA
import altair as alt
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.spatial.distance import cosine


def find_optimal_num_clusters(embeddings):
    kmeans_inertia = []

    #try num_clusters from [3, 29]
    k_clusters = np.arange(2,25)

    for n_cluster in k_clusters:

        #sklearn implementation
        kmeans = KMeans(n_clusters = n_cluster, random_state = 42)

        #fit model and return labels
        labels = kmeans.fit_predict(embeddings)

        inertia = kmeans.inertia_
        kmeans_inertia.append(inertia)

    kmeans_inertia = pd.Series(data = kmeans_inertia, name = 'inertia')
    number_of_clusters = pd.Series(data = range(2,25, 1), name = 'number of clusters')
    fig = go.Figure(data = go.Scatter(x=number_of_clusters, 
                                      y=kmeans_inertia,
                                      mode='lines+markers',
                                      marker=dict(
                                                symbol='circle',    
                                                size=10,           
                                                color='#143642'),      

                                      line=dict(
                                                color='#0f8b8d',        
                                                width=2)))
    
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            )
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            )
        ),
        autosize=False,
        width = 1000,
        height = 600,
        xaxis_title='Inertia',
        yaxis_title='Number of Clusters'
    )
    fig.update_yaxes(title_font_color='black')
    fig.update_xaxes(title_font_color='black')
    return fig

st.set_page_config(page_title="Clustering")
#change the labels to be a string instead of number
dataset = pd.read_json('data/errors.json')
model = SentenceTransformer('all-distilroberta-v1')

embeddings = model.encode(dataset.text)
reduced_embeddings = PCA(n_components=10).fit_transform(embeddings)
X_embedded = TSNE(n_components=2).fit_transform(embeddings)
embeddings = embeddings.tolist()
reduced_embeddings = reduced_embeddings.tolist()

dataset = pd.DataFrame({
    'embedding': embeddings,
    dataset.text.name: dataset.text,
    dataset.label.name: dataset.label,
    dataset.predicted.name: dataset.predicted,
    'reduced_embedding': reduced_embeddings, 
    'x': X_embedded[:,0],
    'y': X_embedded[:,1]
})

st.markdown(
    """
    Typical errors analysis frameworks involve grouping the errors made and try to come up with a label. We have 1776 errors, that will take a long time to manually look through :( We will automate this process by clustering :) 

    We will use [K-Means](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) clustering algorithm to cluster the errors made. The number of clusters is an important hyperparameter when running K-Means. To determine the optimal number of clusters, we will use the elbow method.

    Before can test different number of clusters we need to convert the text to a representation that computer would understand. We will use [Sentence-BERT](https://www.sbert.net/) embeddings of the misclassified examples. Sentence-BERT models produce semantically meaningful sentence embeddings, which means sentences with similar meaning will be closer together in the vector space. 

    There are many Sentence-BERT models available, the particular [model](https://huggingface.co/sentence-transformers/all-distilroberta-v1) we're using produces embeddings with 768 dimensions. 

    We plot the inertia (the sum of squared distances of samples to their closest cluster center) against the number of clusters. The optimal number is at the elbow. 
    """
)

figure = find_optimal_num_clusters(embeddings)
st.plotly_chart(figure)

st.markdown(
    """
    To visualize the examples in a scatter plot we do dimensionality reduction using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to reduce the embeddings to two dimensions.

    If you hover over the individual dots you can see the example, predicted label, and the actual label.  
    If you wish to experiment with the number of clusters yourself, there is a slider below that allows you to change the number of clusters.  
    """
)

slider = st.slider(label = 'number of cluster to show', min_value = 2, max_value = 20, step = 1)
#use embeddings of text to cluster 
embeddings = np.array(dataset.embedding.tolist())
Kmeans_clusterer = KMeans(n_clusters = slider, 
                            random_state = 42)
clusters = Kmeans_clusterer.fit_predict(embeddings)
#new column for the cluster assigned to each row
dataset['cluster'] = pd.Series(clusters, index = dataset.index).astype('int')
#centroid of cluster is the mean of all the points in the cluster
dataset['centroid'] = dataset.cluster.apply(lambda x: Kmeans_clusterer.cluster_centers_[x])

scatter = alt.Chart(dataset).mark_point(size=200, filled=True).encode(
    x = alt.X('x:Q', axis=None),
    y = alt.Y('y:Q', axis=None),
    color = alt.Color('cluster:N',
                      scale=alt.Scale(scheme='category20b')
                      ).legend(direction = 'vertical', 
                               symbolSize = 200, 
                               labelFontSize=14, 
                               titleFontSize=20,
                               titleColor='black',
                               labelColor='black'),
    #TODO change the predicated and label to strings
    tooltip = ['cluster:N', 'text:N',"predicted:N", "label:N"],
    ).properties(
        width = 1000,
        height = 600
    ).interactive()

# save the baseline for now
# idealy you want to use the cache or something
dataset.to_json('data/updated_baseline.json', orient = 'records', indent = 4)

# TODO fix the placing of this
st.altair_chart(scatter)


dataset.loc[dataset['predicted'] == 1.0, 'pred_label'] = 'Positive'
dataset.loc[dataset['predicted'] == 0.0, 'pred_label'] = 'Negative'

fig = alt.Chart(dataset).mark_bar().encode(
    x=alt.X("cluster:N").title('Cluster Number'
                               ).axis(labelColor='black', 
                                      labelFontSize=14, 
                                      titleColor='black'),
    y=alt.Y("count(pred_label)").stack("normalize"
                                       ).title('Precent of Predicted Label'
                                               ).axis(labelColor='black', 
                                                      labelFontSize=14,
                                                      titleColor='black'),
    color=alt.Color('pred_label').scale(range=['red', 'blue']).title("Predicted Label").legend(titleColor='black', labelColor='black')
    ).properties(
            width = 1000,
            height = 600
    )
st.altair_chart(fig)

st.markdown(
    """

    Now we know the examples that were misclassified and which examples are somehow related to each other. But we want to know more about features of the misclassified examples..
    
    In the next visualization we will look at the false positives and false negatives.

    """
)
# TODO save the values and data the user enters so it's not computing stuff every time