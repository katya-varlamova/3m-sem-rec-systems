import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from time import time
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pymorphy2 import MorphAnalyzer
import warnings
from lda import *
from tfidf import *
nltk.download('stopwords')
nltk.download('punkt')
warnings.filterwarnings("ignore")
morph = MorphAnalyzer()
stop_words = set(stopwords.words('english'))

lemma = True
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    if lemma:
        tokens = [morph.parse(token)[0].normal_form for token in tokens]
    return ' '.join(tokens)

name = "IMDB_larger_description_dataset.csv" #sys.argv[1]
base_col_name = "description" #sys.argv[2]
clusers_col_name = "genre" #sys.argv[3]
data = pd.read_csv(name)

##data = {"description" : [],
##        "genre" : []}
##for i in range(10):
##    data["description"].append([f"Документ{i // 2}", f"описания{i // 2}", f"с{i // 2}",
##                 f"Документ{i // 2}", f"описания{i // 2}", f"с{i // 2}",
##                 f"Документ{i // 2}", f"описания{i // 2}", f"с{i // 2}",
##                 f"Документ{i // 2}", f"описания{i // 2}", f"с{i // 2}"])
##    data["genre"].append(f"g{i // 2}")


num_docs = min(len(data), 250)
step = 10
start = 10
errors_lda = []
errors_tfidf_kmeans = []
times_lda = []
times_tfidf_kmeans = []

errors_lda_my = []
errors_tfidf_kmeans_my = []
times_lda_my = []
times_tfidf_kmeans_my = []

for i in range(start, num_docs, step):
    docs = data[base_col_name][:i]
    genres = data[clusers_col_name][:i]
    docs = docs.apply(preprocess_text)
    
    tfidf = TfidfVectorizer()
    lda = LatentDirichletAllocation(n_components=len(set(genres)), random_state=42)
    kmeans = KMeans(n_clusters=len(set(genres)), random_state=42)

    
    # LDA
    start_time = time()
    count_vectorizer = CountVectorizer()
    count_data = count_vectorizer.fit_transform(docs).toarray()
    lda = LatentDirichletAllocation(n_components=len(set(genres)), random_state=0)
    lda_matrix = lda.fit_transform(count_data)
    lda_clusters = [list(row).index(max(row)) for row in lda_matrix]
    time_lda = time() - start_time

    # LDA my
    start_time = time()
    count_vectorizer = CountVectorizer()
    count_data = count_vectorizer.fit_transform(docs).toarray()
    topic_distribution = lda_func(count_data, len(set(genres)), 0.1, 0.1, 10)
    lda_clusters_my = [list(row).index(max(row)) for row in topic_distribution]
    #print(lda_clusters)
    time_lda_my = time() - start_time


    # TFIDF + KMeans
    start_time = time()
    tfidf_matrix = tfidf.fit_transform(docs)
    kmeans_pipe = make_pipeline(Normalizer(), kmeans)
    kmeans_clusters = kmeans_pipe.fit_predict(tfidf_matrix)
    time_tfidf_kmeans = time() - start_time

    # TFIDF + KMeans my
    start_time = time()
    tfidf_my = TFIDF()
    tfidf_my.add_documents(docs)
    matrix = tfidf_my.calculate_tfidf()
    kmeans_pipe = make_pipeline(Normalizer(), kmeans)
    kmeans_clusters_my = kmeans_pipe.fit_predict(matrix)
    time_tfidf_kmeans_my = time() - start_time

    error_lda = adjusted_rand_score(genres, lda_clusters)
    error_tfidf_kmeans = adjusted_rand_score(genres, kmeans_clusters)

    error_lda_my = adjusted_rand_score(genres, lda_clusters_my)
    error_tfidf_kmeans_my = adjusted_rand_score(genres, kmeans_clusters_my)
    
    errors_lda_my.append(error_lda_my)
    errors_tfidf_kmeans_my.append(error_tfidf_kmeans_my)
    times_lda_my.append(time_lda_my)
    times_tfidf_kmeans_my.append(time_tfidf_kmeans_my)

    errors_lda.append(error_lda)
    errors_tfidf_kmeans.append(error_tfidf_kmeans)
    times_lda.append(time_lda)
    times_tfidf_kmeans.append(time_tfidf_kmeans)
    print(error_lda, error_tfidf_kmeans, "done")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(range(start, num_docs, step), errors_lda, label='LDA')
plt.plot(range(start, num_docs, step), errors_tfidf_kmeans, label='TFIDF + KMeans')
plt.plot(range(start, num_docs, step), errors_lda_my, label='LDA (my)')
plt.plot(range(start, num_docs, step), errors_tfidf_kmeans_my, label='TFIDF (my) + KMeans')
plt.xlabel('Number of Documents')
plt.ylabel('Accuracy')
plt.title('Clustering accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(start, num_docs, step), times_lda, label='LDA')
plt.plot(range(start, num_docs, step), times_tfidf_kmeans, label='TFIDF + KMeans')
plt.plot(range(start, num_docs, step), times_lda_my, label='LDA (my)')
plt.plot(range(start, num_docs, step), times_tfidf_kmeans_my, label='TFIDF (my) + KMeans')
plt.xlabel('Number of Documents')
plt.ylabel('Time (s)')
plt.title('Clustering Time')
plt.legend()

plt.tight_layout()
plt.show()
plt.savefig("res.png")
