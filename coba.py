import math

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.doc_term_freqs = [self.get_term_freqs(doc) for doc in documents]
        self.total_docs = len(documents)
        self.doc_freqs = self.get_doc_freqs()
        self.idf = self.calculate_idf()

    def get_term_freqs(self, document):
        term_freqs = {}
        for term in document.split():
            term_freqs[term] = term_freqs.get(term, 0) + 1
        return term_freqs

    def get_doc_freqs(self):
        doc_freqs = {}
        for doc_term_freq in self.doc_term_freqs:
            for term in doc_term_freq:
                doc_freqs[term] = doc_freqs.get(term, 0) + 1
        return doc_freqs

    def calculate_idf(self):
        idf = {}
        for term, doc_freq in self.doc_freqs.items():
            idf[term] = math.log((self.total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
        return idf

    def calculate_bm25_score(self, query, doc_index):
        score = 0
        doc_term_freq = self.doc_term_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index]
        for term in query.split():
            if term in doc_term_freq:
                df = self.doc_freqs[term]
                idf = self.idf[term]
                tf = doc_term_freq[term]
                score += (idf * tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length)))
        return score

    def rank_documents(self, query):
        scores = [(i, self.calculate_bm25_score(query, i)) for i in range(self.total_docs)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# Contoh penggunaan
documents = [
    "Python is a popular programming language.",
    "BM25 is an algorithm used in information retrieval.",
    "Information retrieval is the process of obtaining information from a large repository."
]

bm25 = BM25(documents)
query = "Python algorithm information"
results = bm25.rank_documents(query)

print("Hasil Pencarian:")
for rank, (doc_index, score) in enumerate(results, start=1):
    print(f"{rank}. Dokumen {doc_index + 1} - Score: {score}")
