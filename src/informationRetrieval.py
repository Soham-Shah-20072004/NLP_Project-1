import math
from collections import defaultdict, Counter
from .util import *

class InformationRetrieval():

    def __init__(self):
        self.index = None
        self.idf = {}
        self.doc_vecs = [] 
        self.doc_norms = []
        self.doc_ids = []

    def buildIndex(self, docs, docIDs):

        index = None
        N = len(docs)
        self.doc_ids = docIDs
        doc_tfs = []
        df = defaultdict(int)
        for doc in docs:
            terms = []
            for sentence in doc:
                for word in sentence:
                    terms.append(word)

            tf = Counter(terms)
            doc_tfs.append(tf)
            for term in tf:
                df[term] += 1

        self.idf = {}

        for term in df:
            self.idf[term] = math.log10(N / df[term])   # idf formula

        self.doc_vecs = []
        self.doc_norms = []
        for tf in doc_tfs:
            vec = {}
            norm = 0
            for term, count in tf.items():
                tfidf = count * self.idf[term]  # tfidf
                vec[term] = tfidf
                norm += tfidf * tfidf   # doc l2 norm 

            self.doc_vecs.append(vec)
            self.doc_norms.append(math.sqrt(norm))

        index = {
            "idf": self.idf,
            "doc_vecs": self.doc_vecs,
            "doc_norms": self.doc_norms,
            "doc_ids": self.doc_ids
        }
        self.index = index

    def rank(self, queries):
        res = []
        for query in queries:
            terms = []
            for sentence in query:
                for term in sentence:
                    terms.append(term)
            q_tf = Counter(terms)
            qvec = {}
            qnorm = 0

            for term, count in q_tf.items():
                if term in self.idf:            # ignore terms not in vocab
                    tfidf = count * self.idf[term]
                    qvec[term] = tfidf
                    qnorm += tfidf * tfidf
            qnorm = math.sqrt(qnorm)
            scores = []

            for idx, doc_vec in enumerate(self.doc_vecs):
                doc_norm = self.doc_norms[idx]
                if qnorm == 0 or doc_norm == 0:
                    score = 0
                else:
                    dot = 0
                    for term in qvec:
                        if term in doc_vec:
                            dot += qvec[term] * doc_vec[term]
                    score = dot / (qnorm * doc_norm)

                scores.append((score, self.doc_ids[idx]))

            scores.sort(key=lambda x: x[0], reverse=True)
            ranked_ids = [doc_id for score, doc_id in scores]

            res.append(ranked_ids)
        return res
