from keras_search_engine_web.wordvec_glove_feature_extractor import WordVecGloveFeatureExtractor
import numpy as np

GLOVE_EMBEDDING_SIZE = 100


class DocSearchEngine(object):
    fe = None
    doc_features = None
    doc_paths = []

    def __init__(self):
        self.fe = WordVecGloveFeatureExtractor()
        self.doc_features = []

    def index_document(self, doc_text):
        doc_feature = self.fe.extract(doc_text)
        doc_id = len(self.doc_features)
        self.doc_features.append(doc_feature)
        self.doc_paths[doc_id] = doc_text

    def index_documents(self, doc_texts):
        if len(self.doc_features) > 0:
            self.doc_features = self.doc_features + self.fe.extract_all(doc_texts)
        else:
            self.doc_features = self.fe.extract_all(doc_texts)

    def get_top_k(self, query, k):
        query_feature = self.fe.extract(query)
        doc_count = len(self.doc_features)
        query_diff = []
        for doc_id in range(doc_count):
            doc_feature = self.doc_features[doc_id]
            query_diff.append(doc_feature - query_feature)
        dist = np.linalg.norm(query_diff, axis=1)
        ids = np.argsort(dist)[:min(len(dist), k)]
        dist = [dist[id] for id in ids]
        return ids, dist

    def get_doc(self, doc_ids):
        return [self.doc_paths[doc_id] for doc_id in doc_ids]

    def test_run(self):
        self.index_documents(['Hello World', 'What a wonderful world!'])
        print(self.get_top_k('Hello', k=30))


if __name__ == '__main__':
    search_engine = DocSearchEngine()
    search_engine.test_run()
