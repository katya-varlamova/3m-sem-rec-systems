import math
import numpy as np
class TFIDF:
    def __init__(self):
        self.documents = []
        self.term_index = {}
        self.num_documents = 0
        self.tf = {}
        self.df = {}

    def add_documents(self, documents):
        self.documents = documents
        self.num_documents = len(documents)
        self._calculate_frequencies()

    def _calculate_frequencies(self):
        for doc_index, doc in enumerate(self.documents):
            unique_terms = set(doc)
            for term in doc:
                if (doc_index, term) not in self.tf:
                    self.tf[(doc_index, term)] = 0
                self.tf[(doc_index, term)] += 1

            for term in unique_terms:
                if term not in self.df:
                    self.df[term] = 0
                self.df[term] += 1

                if term not in self.term_index:
                    self.term_index[term] = len(self.term_index)

    def calculate_tfidf(self):
        tfidf = np.zeros([self.num_documents, len(self.term_index)])
        for (doc_index, term), tf_value in self.tf.items():
            term_index = self.term_index[term]
            
            idf = math.log(self.num_documents / self.df[term]) + 1
            tfidf[doc_index][term_index] = tf_value * idf
        return tfidf


def ex():
    documents = [
        ["это", "пример"],
        ["это", "еще", "один", "документ"],
        ["документ", "содержит", "текст"]
    ]

    tfidf = TFIDF()
    tfidf.add_documents(documents)
    matrix = tfidf.calculate_tfidf()
    print(matrix)

