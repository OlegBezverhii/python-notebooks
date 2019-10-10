# coding: utf-8
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3
import matplotlib.pyplot as plt
import matplotlib as mpl

titles = open('material4.csv', 'r', encoding='utf-8', errors='ignore').read().split('\n')
print(str(len(titles)) + ' запросов считано')

stopwords = nltk.corpus.stopwords.words('russian')
stopwords.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', 'к', 'на'])

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")

def token_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def token_only(text):
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[а-яА-Я]', token):
            filtered_tokens.append(token)
    return filtered_tokens

totalvocab_stem = []
totalvocab_token = []
for i in titles:
    allwords_stemmed = token_and_stem(i)
    #print(allwords_stemmed)
    totalvocab_stem.extend(allwords_stemmed)
    
    allwords_tokenized = token_only(i)
    totalvocab_token.extend(allwords_tokenized)


import pymorphy2
morph = pymorphy2.MorphAnalyzer()

G=[]

for i in titles:
    h=i.split(' ')
    #print(h)
    s=''
    for k in h:
        #print(k)
        p = morph.parse(k)[0].normal_form
        #print(p)
        s+=' '
        s += p
        #print(s)
        #G.append(p)
    #print(s)
    G.append(s)

pymof = open('pymof_pod.txt', 'w', encoding='utf-8', errors='ignore')
pymofcsv = open('pymofcsv_pod.csv', 'w', encoding='utf-8', errors='ignore')
for item in G:
    pymof.write("%s\n" % item)
    pymofcsv.write("%s\n" % item)
pymof.close()
pymofcsv.close()

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

n_featur=200000
tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000,
                                 min_df=0.01, stop_words=stopwords,
                                 use_idf=True, tokenizer=token_and_stem, ngram_range=(1,3))
get_ipython().magic('time tfidf_matrix = tfidf_vectorizer.fit_transform(titles)')
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)
dist.shape


#Word2Vec

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
train_data_features = vectorizer.fit_transform(titles)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names()

from gensim.models import Word2Vec
from scipy.sparse import csr as _csr

num_features = 300    # Размерность верктора слова
min_word_count = 20   # Минимальное количество слов
num_workers = 2       # Количество потоков для выполнения параллельно
context = 3          # Размер окна контекста 
downsampling = 1e-3   # Downsample установка для частых слов

print ("Training Word2Vec model...")
modelW2V = Word2Vec(G, workers=num_workers,             size=num_features, min_count = min_word_count,             window = context, sample = downsampling, seed=1)

modelW2V.init_sims(replace=True)

model_name = "searchcluster"
modelW2V.save(model_name)

modelW2V.index2word

modelW2V.most_similar(positive=['министерство', 'обороны'])

modelW2V.syn0.shape

from sklearn.metrics.pairwise import cosine_similarity
distw2v = 1 - cosine_similarity(modelW2V.syn0)
distw2v.shape

from sklearn.manifold import MDS

MDS()

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(distw2v)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


color_list = generate_colors(num_clusters)
cluster_colors = {}
for i in range(0,num_clusters):
    cluster_colors[i] = color_list[i]


get_ipython().magic('matplotlib inline')
fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(xs, ys, marker='o', linestyle='', ms=12, mec='none')
plt.show() 

tfidf_matrix = modelW2V.syn0

from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(40, 15)) # set size
ax = dendrogram(linkage_matrix, orientation="top");

plt.tick_params(     axis= 'x',          # changes apply to the x-axis
     which='both',      # both major and minor ticks are affected
     bottom='off',      # ticks along the bottom edge are off
     top='off',         # ticks along the top edge are off
     labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.savefig('w2v_clusters.png', dpi=200)


# In[69]:

#создаем data frame, который содержит PCA + номера кластеров и сами запросы
df = pd.DataFrame(dict(x=xs, y=ys, label=clusterkm)) 
#группируем по кластерам
groups = df.groupby('label')

dfsvd = pd.DataFrame(dict(x=xs, y=ys, label=clustersvd)) 

groups_svd = dfsvd.groupby('label')

dfmbk = pd.DataFrame(dict(x=xs, y=ys, label=clustermbk)) 

groups_mbk = dfmbk.groupby('label')


# In[77]:

fig, ax = plt.subplots(figsize=(30, 15)) #figsize=(51, 25)

for name, group in groups_mbk:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',         
        which='both',     
        bottom='off',      
        top='off',         
        labelbottom='off')
    ax.tick_params(        axis= 'y',         
        which='both',     
        left='off',      
        top='off',
        labelleft='off')
    
ax.legend(numpoints=1)  #показать легенду только 1 точки

#добавлим метки/названия в х,у позиции с поисковым запросом
#for i in range(len(df)):
#    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

plt.show()
plt.savefig('w2v_clusters_kmeans.png', dpi=200)


# Метод к-средних - KMeans

from sklearn.cluster import KMeans

num_clusters = 5#25
km = KMeans(n_clusters=num_clusters)
get_ipython().magic('time km.fit(tfidf_matrix)')
idx = km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

print(clusters)
print (km.labels_)


# # SVD

from sklearn.decomposition import TruncatedSVD

modelkmen = KMeans(n_clusters=num_clusters)#, random_state=42)
svd_init = TruncatedSVD() #(n_components=1000, random_state=123)
features = svd_init.fit_transform(tfidf_matrix)
#print(features)
svd = modelkmen.fit_predict(features)
#print(svd)


# MiniBatchKMeans

from sklearn.cluster import MiniBatchKMeans
#kol_clusters = 25
mbk  = MiniBatchKMeans(init='random', n_clusters=num_clusters) #(init='k-means++', ‘random’ or an ndarray)
mbk.fit_transform(tfidf_matrix)
%time mbk.fit(tfidf_matrix)
miniclusters = mbk.labels_.tolist()
print (mbk.labels_)


# DBSCAN

from sklearn.cluster import DBSCAN

get_ipython().magic('time db = DBSCAN(eps=0.3, min_samples=10).fit(tfidf_matrix)')
labels = db.labels_
labels.shape
print(labels)

# Аггломеративная класстеризация

from sklearn.cluster import AgglomerativeClustering

#agglo = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='complete')
#answer = agglo.fit_predict(tfidf_matrix.toarray())
#answer.shape

agglo1 = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean')
get_ipython().magic('time answer1 = agglo1.fit_predict(tfidf_matrix.toarray())')
answer1.shape

agglo2 = AgglomerativeClustering(n_clusters=num_clusters, affinity='l1', linkage='complete')
get_ipython().magic('time answer2 = agglo2.fit_predict(tfidf_matrix.toarray())')
answer2.shape

agglo3 = AgglomerativeClustering(n_clusters=num_clusters, affinity='l2', linkage='complete')
get_ipython().magic('time answer3 = agglo3.fit_predict(tfidf_matrix.toarray())')
answer3.shape

agglo4 = AgglomerativeClustering(n_clusters=num_clusters, affinity='manhattan', linkage='complete')
get_ipython().magic('time answer4 = agglo4.fit_predict(tfidf_matrix.toarray())')
answer4.shape


 import pickle
 with open("4result_agglo.pkl", 'wb') as handle:
                     pickle.dump(answer, handle)
 import pickle
 with open("4result_agglo1.pkl", 'wb') as handle:
                     pickle.dump(answer1, handle)
 import pickle
 with open("4result_agglo2.pkl", 'wb') as handle:
                     pickle.dump(answer2, handle)
 import pickle
 with open("4result_agglo3.pkl", 'wb') as handle:
                     pickle.dump(answer3, handle)
 import pickle
 with open("4result_agglo4.pkl", 'wb') as handle:
                     pickle.dump(answer4, handle)

#k-means
clusterkm = km.labels_.tolist()
#svd
clustersvd = svd
#minikmeans
clustermbk = mbk.labels_.tolist()


#dbscan
clusters3 = labels
 
#agglo
#clusters4 = answer.tolist()

clusters5 = answer1.tolist()
clusters6 = answer2.tolist()
clusters7 = answer3.tolist()
clusters8 = answer4.tolist()

frame = pd.DataFrame(titles, index = [clusterkm])

#k-means
out = { 'title': titles, 'cluster': clusterkm }
frame1 = pd.DataFrame(out, index = [clusterkm], columns = ['title', 'cluster'])

#svd
out = { 'title': titles, 'cluster': clustersvd }
frame_svd = pd.DataFrame(out, index = [clustersvd], columns = ['title', 'cluster'])

#mini
out = { 'title': titles, 'cluster': clustermbk }
frame_minik = pd.DataFrame(out, index = [clustermbk], columns = ['title', 'cluster'])

print(frame_svd)

#dbscan
# out = { 'title': titles, 'cluster': clusters3 }
# frame_dbscan = pd.DataFrame(out, index = [clusters3], columns = ['title', 'cluster'])
# #agglo
# out = { 'title': titles, 'cluster': clusters4 }
# frame_agglo = pd.DataFrame(out, index = [clusters4], columns = ['title', 'cluster'])
# out = { 'title': titles, 'cluster': clusters5 }
# frame_agglo1 = pd.DataFrame(out, index = [clusters5], columns = ['title', 'cluster'])
# out = { 'title': titles, 'cluster': clusters6 }
# frame_agglo2 = pd.DataFrame(out, index = [clusters6], columns = ['title', 'cluster'])
# out = { 'title': titles, 'cluster': clusters7 }
# frame_agglo3 = pd.DataFrame(out, index = [clusters7], columns = ['title', 'cluster'])
# out = { 'title': titles, 'cluster': clusters8 }
# frame_agglo4 = pd.DataFrame(out, index = [clusters8], columns = ['title', 'cluster'])

# print(frame)
# #print(frame_svd)
# #print(frame_minik)
# #print(frame_dbscan)

# frame1['cluster'].value_counts()
# frame_svd['cluster'].value_counts()
# frame_minik['cluster'].value_counts()

# frame_dbscan['cluster'].value_counts()
# frame_agglo['cluster'].value_counts()
# frame_agglo1['cluster'].value_counts()
# frame_agglo2['cluster'].value_counts()
# frame_agglo3['cluster'].value_counts()
# frame_agglo4['cluster'].value_counts()

terms = tfidf_vectorizer.get_feature_names()
vocab_frame = pd.DataFrame({'слова': totalvocab_token}, index = totalvocab_stem)


from __future__ import print_function

print("Топ 10 термов в данном кластере:")
print()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#print(order_centroids)
#print(num_clusters)
for i in range(num_clusters):
    print("Кластер %d. Слова :" % i, end='')
    for ind in order_centroids[i, :9]:
        print(' %s' % vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0], end=',')
    print()
    print()
    print("Кластер %d. Запросы:" % i, end='')
    for title in frame1.ix[i]['title'].values.tolist():
        print(' %s,' % title, end='')
    print()
    print()


# Метод главных компонент - PCA

from sklearn.decomposition import IncrementalPCA
icpa = IncrementalPCA(n_components=2, batch_size=16)
get_ipython().magic('time icpa.fit(dist) #demo =')
get_ipython().magic('time demo2 = icpa.transform(dist)')

xs, ys = demo2[:, 0], demo2[:, 1]

# PCA 3D
from sklearn.decomposition import IncrementalPCA
icpa = IncrementalPCA(n_components=3, batch_size=16)
get_ipython().magic('time icpa.fit(dist) #demo =')
get_ipython().magic('time ddd = icpa.transform(dist)')
xs, ys, zs = ddd[:, 0], ddd[:, 1], ddd[:, 2]


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# Визуализация кластеров документов

from matplotlib import rc
#включаем русские символы на графике
font = {'family' : 'Verdana'}#, 'weigth': 'normal'}
rc('font', **font)

from nltk.tag import pos_tag

def strip_proppers_POS(text):
    tagged = pos_tag(text.split())
    non_propernouns = [word for word,pos in tagged if pos != 'NNP' and pos != 'NNPS']
    return non_propernouns

import random
def generate_colors(n):
    color_list = []
    for c in range(0,n):
        r = lambda: random.randint(0,255)
        color_list.append( '#%02X%02X%02X' % (r(),r(),r()) )
    return color_list


#устанавливаем цвета
cluster_colors = {0: '#ff0000',
                  1: '#ff0066',
                  2: '#ff0099',
                  3: '#ff00cc',
                  4: '#ff00ff',
                  5: '#cc00ff',
                  6: '#6600ff',
                  7: '#3300ff',
                  8: '#0033ff',
                  9: '#0066ff',
                  10: '#0099ff',
                  11: '#00ccff',
                  12: '#339999',
                  13: '#00ffff',
                  14: '#00ffcc',
                  15: '#00ff99',
                  16: '#00ff66',
                  17: '#00ff33',
                  18: '#00ff00',
                  19: '#33ff00',
                  20: '#66ff00',
                  21: '#99ff00',
                  22: '#ccff00',
                  23: '#ffff00',
                  24: '#ffcc00',
                  25: '#ff9900',
                 }

#даем имена кластерам
cluster_names = {0: '0', 
                 1: '1',
                 2: '2',
                 3: '3',
                 4: '4',
                 5: '5',
                 6: '6',
                 7: '7',
                 8: '8',
                 9: '9',
                 10: '10',
                 11: '11',
                 12: '12',
                 13: '13',
                 14: '14',
                 15: '15',
                 16: '16',
                 17: '17',
                 18: '18',
                 19: '19',
                 20: '20',
                 21: '21',
                 22: '22',
                 23: '23',
                 24: '24',
                 25: '25',
                }

get_ipython().magic('matplotlib inline')


#создаем data frame, который содержит PCA + номера кластеров и сами запросы
df = pd.DataFrame(dict(x=xs, y=ys, label=clusterkm, title=titles)) 
#группируем по кластерам
groups = df.groupby('label')

dfsvd = pd.DataFrame(dict(x=xs, y=ys, label=clustersvd, title=titles)) 

groups_svd = dfsvd.groupby('label')

dfmbk = pd.DataFrame(dict(x=xs, y=ys, label=clustermbk, title=titles)) 

groups_mbk = dfmbk.groupby('label')

# groups.head()

df.to_csv(path_or_buf="kmean.csv",sep=',', index=False)
df.to_json(path_or_buf="kmean.json")

dfsvd.to_csv(path_or_buf="svd.csv",sep=',', index=False)

dfmbk.to_csv(path_or_buf="mbk.csv",sep=',', index=False)

fig, ax = plt.subplots(figsize=(72, 36)) #figsize=(51, 25)

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label=cluster_names[name], color=cluster_colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(        axis= 'x',          
        which='both',      
        bottom='off',      
        top='off',         
        labelbottom='off')
    ax.tick_params(        axis= 'y',         
        which='both',     
        left='off',      
        top='off',       
        labelleft='off')
    
ax.legend(numpoints=1)  #показать легенду только 1 точки

#добавлим метки/названия в х,у позиции с поисковым запросом
#for i in range(len(df)):
#    ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)  

#показать график
plt.show() 
plt.close()

css = """
text.mpld3-text, div.mpld3-tooltip {
  font-family:Arial, Helvetica, sans-serif;
}

g.mpld3-xaxis, g.mpld3-yaxis {
display: none; }

svg.mpld3-figure {
margin-left: -400px;}
"""

css = ""

# Plot 
fig, ax = plt.subplots(figsize=(25,27)) 
ax.margins(0.03)


for name, group in groups_mbk: #groups_svd groups_mbk
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, #ms=18
                     label=cluster_names[name], mec='none', 
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]
    
    tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, #css=css)
                                       voffset=10, hoffset=10, css=css)
    mpld3.plugins.connect(fig, tooltip) #   , TopToolbar()
    
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    #ax.axes.get_xaxis().set_visible(False)
    #ax.axes.get_yaxis().set_visible(False)

ax.set_title("Mini K-Means", size=20) #groups_mbk
    
ax.legend(numpoints=1)
mpld3.disable_notebook()
#mpld3.display()
mpld3.save_html(fig, "mbk.html")

mpld3.show()mpld3.save_json(fig, "vivod.json")mpld3.fig_to_html(fig)#fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
fig, ax = plt.subplots(figsize=(51,25))

scatter = ax.scatter(np.random.normal(size=N),
                     np.random.normal(size=N),
                     c=np.random.random(size=N),
                     s=1000 * np.random.random(size=N),
                     alpha=0.3,
                     cmap=plt.cm.jet)
ax.grid(color='white', linestyle='solid')    

ax.set_title("Кластеры", size=20)

fig, ax = plt.subplots(figsize=(51,25))

labels = ['point {0}'.format(i + 1) for i in range(N)]
tooltip = mpld3.plugins.PointLabelTooltip(scatter, labels=labels)
mpld3.plugins.connect(fig, tooltip)

mpld3.show()fig, ax = plt.subplots(figsize=(72,36))
for name, group in groups:
    points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=18, 
                     label=cluster_names[name], mec='none', 
                     color=cluster_colors[name])
    ax.set_aspect('auto')
    labels = [i for i in group.title]
    tooltip = mpld3.plugins.PointLabelTooltip(points, labels=labels)
    mpld3.plugins.connect(fig, tooltip)

ax.set_title("K-means", size=20)

mpld3.display()

# # Иерархическая кластеризация документов 

from scipy.cluster.hierarchy import ward, dendrogram

get_ipython().magic('time linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances')

fig, ax = plt.subplots(figsize=(27, 15)) 
ax = dendrogram(linkage_matrix, orientation="top"); #labels=titles

plt.tick_params(    axis= 'x',          
    which='both',    
    bottom='off',
    #top='off',
    labelbottom='off')

plt.tight_layout()
namefile = 'ierarhik.png'
plt.savefig(namefile, dpi=200)
plt.close()


import plotly
plotly.__version__

import plotly.plotly as py
import plotly.graph_objs as go

trace1 = go.Scatter3d(
    x=xs,
    y=ys,
    z=zs,
    mode='markers',
    marker=dict(
        size=12,
        line=dict(
            color='rgba(217, 217, 217, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='cluster-3d-plot')

import vk
#передаешь id сессии
session = vk.Session(access_token='')
# URL для получения access_token:
# https://oauth.vk.com/authorize?client_id=tvoiid&scope=friends,pages,groups,offline&redirect_uri=https://oauth.vk.com/blank.html&display=page&v=5.21&response_type=token
api = vk.API(session)
poss=[]

id_pab=-59229916
info=api.wall.get(owner_id=id_pab, offset=0, count=1)

kolvo = (info[0]//100)+1

shag=100
sdvig=0
h=0

import time

while h<kolvo:
    if(h>70):
        print(h)
    pubpost=api.wall.get(owner_id=id_pab, offset=sdvig, count=100)
    i=1
    while i < len(pubpost):
        b=pubpost[i]['text']
        poss.append(b)
        i=i+1
    h=h+1
    sdvig=sdvig+shag
    time.sleep(1)
len(poss)

import io
with io.open("podslyshanobgpu.txt", 'w', encoding='utf-8', errors='ignore') as file:
    for line in poss:
        file.write("%s\n" % line)
file.close()

titles = open('podslyshanobgpu.txt', encoding='utf-8', errors='ignore').read().split('\n')

print(str(len(titles)) + ' постов считано')

import re

posti=[]

for line in titles:
    chis = re.sub(r'(\<(/?[^>]+)>)', ' ', line)
    #chis = re.sub()
    chis = re.sub('[^а-яА-Я ]', '', chis)
    posti.append(chis)