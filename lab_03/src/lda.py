import random
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
def lda_func(data, num_topics, alpha, beta, num_iterations):
    num_docs, num_words = data.shape
    num_tokens = data.sum()

    # Инициализация тем
    topics = np.random.randint(num_topics, size=(num_docs, num_words))

    # Подсчет начальных значений topic_word_counts и doc_topic_counts
    topic_word_counts = np.zeros((num_topics, num_words))
    doc_topic_counts = np.zeros((num_docs, num_topics))

    for i in range(num_docs):
        for j in range(num_words):
            count = data[i, j]
            if count > 0:
                topic_word_counts[topics[i, j], j] += count
                doc_topic_counts[i, topics[i, j]] += count

    for _ in range(num_iterations):
        for i in range(num_docs):
            for j in range(num_words):
                count = data[i, j]
                if count > 0:
                    word_topic = topics[i, j]
                    topic_word_counts[word_topic, j] -= count
                    doc_topic_counts[i, word_topic] -= count

                    # Вычисление вероятностей тем
                    topic_probs = (doc_topic_counts[i] + alpha) * (topic_word_counts[:, j] + beta) / (num_tokens * beta + topic_word_counts.sum(axis=1))

                    # Нормализация вероятностей
                    topic_probs /= topic_probs.sum()

                    # Выбор новой темы
                    new_topic = np.random.choice(num_topics, p=topic_probs)
                    topics[i, j] = new_topic

                    # Обновление счетчиков
                    topic_word_counts[new_topic, j] += count
                    doc_topic_counts[i, new_topic] += count

    # Вычисление распределения тем
    topic_distribution = doc_topic_counts / doc_topic_counts.sum(axis=1, keepdims=True)

    return topic_distribution

def ex():
    np.random.seed(0)
    data = []
    for i in range(10):
        data.append([f"Документ{i // 2}", f"описания{i // 2}", f"с{i // 2}",
                     f"Документ{i // 2}", f"описания{i // 2}", f"с{i // 2}",
                     f"Документ{i // 2}", f"описания{i // 2}", f"с{i // 2}",
                     f"Документ{i // 2}", f"описания{i // 2}", f"с{i // 2}"])
    count_vectorizer = CountVectorizer()
    count_data = count_vectorizer.fit_transform([' '.join(d) for d in data]).toarray()
    
    print(count_data)

    lda = LatentDirichletAllocation(n_components=5, random_state=0)
    lda_matrix = lda.fit_transform(count_data)
    print(lda_matrix)

    topic_distribution = lda_func(count_data, 5, 0.1, 0.1, 1000)
    print(topic_distribution)
#ex()
