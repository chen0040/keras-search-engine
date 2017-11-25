from keras_search_engine_web.glove_feature_extractor import WordVecGloveFeatureExtractor
import numpy as np
import os

GLOVE_EMBEDDING_SIZE = 100
DATA_DIR_PATH = './data/texts'


def compute_cosine_similarity(X, y):
    dist = []
    for doc_id in range(len(X)):
        x = X[doc_id]
        dist.append(np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2)))
    return dist


def compute_similarity(X, y):
    query_diff = []
    doc_count = len(X)
    for doc_id in range(doc_count):
        doc_feature = X[doc_id]
        query_diff.append(doc_feature - y)
    dist = np.linalg.norm(query_diff, axis=1)
    return dist


def get_content(path):
    with open(path, 'r', encoding='utf8') as file:
        return file.read()


class GloveDocSearchEngine(object):
    fe = None
    doc_features = None
    doc_paths = dict()
    use_cosine_distance = True

    def __init__(self):
        self.fe = WordVecGloveFeatureExtractor()
        self.doc_features = []
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

    def index_document(self, doc_text, doc_path=None):
        doc_feature = self.fe.extract(doc_text)
        doc_id = len(self.doc_features)
        self.doc_features.append(doc_feature)
        if doc_path is not None:
            self.doc_paths[doc_id] = doc_path
        else:
            doc_path = './uploads/' + str(doc_id) + '.txt'
            f = open(doc_path, 'wt')
            f.write(doc_text)
            f.close()
            self.doc_paths[doc_id] = doc_path
        return doc_feature.tolist()

    def index_documents(self, doc_texts, doc_paths=None):
        doc_count = len(self.doc_features)
        if len(self.doc_features) > 0:
            self.doc_features = self.doc_features + self.fe.extract_all(doc_texts)
        else:
            self.doc_features = self.fe.extract_all(doc_texts)

        for i in range(len(doc_texts)):
            if doc_paths is not None:
                self.doc_paths[doc_count + i] = doc_paths[i]
            else:
                doc_id = doc_count + i
                doc_text = doc_texts[i]
                doc_path = './uploads/' + str(doc_id) + '.txt'
                f = open(doc_path, 'wt')
                f.write(doc_text)
                f.close()
                self.doc_paths[doc_id] = doc_path

    def rank_top_k(self, query, k=None):
        if k is None:
            k = 10
        query_feature = self.fe.extract(query)
        if self.use_cosine_distance:
            dist = compute_cosine_similarity(self.doc_features, query_feature)
        else:
            dist = compute_similarity(self.doc_features, query_feature)
        ids = np.argsort(dist)[:min(len(dist), k)]
        dist = [dist[id] for id in ids]
        return ids, dist

    def get_doc(self, doc_ids):
        return [self.doc_paths[doc_id] for doc_id in doc_ids]

    def query_top_k(self, query, k=None):
        if k is None:
            k = 10
        ids, dist = self.rank_top_k(query, k)
        result = []
        for i in range(len(ids)):
            doc_id = ids[i]
            rank = dist[i]
            doc_path = self.doc_paths[doc_id]
            result.append({
                'doc_id': doc_id,
                'rank': rank,
                'path': doc_path,
                'content': get_content(doc_path)
            })
        return result

    def do_default_indexing(self):
        for file in os.listdir(DATA_DIR_PATH):
            file_path = os.path.join(DATA_DIR_PATH, file)
            if os.path.isfile(file_path):
                print('processing file: ', file)
                lines = open(file_path, 'rt', encoding='utf8').read().split('\n')
                text = ''
                for line in lines:
                    text += line
                self.index_document(text, file_path)

    def doc_count(self):
        return len(self.doc_features)

    def test_run(self):
        if self.doc_count() == 0:
            self.index_documents(['Hello World', 'What a wonderful world!'])
        print(self.rank_top_k('Hello', k=30))


def main():
    search_engine = GloveDocSearchEngine()
    search_engine.test_run()


if __name__ == '__main__':
    main()
