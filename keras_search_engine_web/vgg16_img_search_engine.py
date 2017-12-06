from keras_search_engine_web.vgg16_feature_extractor import VGG16FeatureExtractor
import numpy as np
import os

DATA_DIR_PATH = './data/images'


class VGG16ImageSearchEngine(object):
    fe = None
    img_features = None
    img_paths = dict()

    def __init__(self):
        self.fe = VGG16FeatureExtractor()
        self.img_features = []
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

    def index_image(self, img_path):
        img_feature = self.fe.extract(img_path)
        img_id = len(self.img_features)
        self.img_features.append(img_feature)
        self.img_paths[img_id] = img_path
        return img_feature.tolist()

    def index_images(self, img_paths):
        img_count = len(self.img_features)
        if len(self.img_features) > 0:
            self.img_features = self.img_features + self.fe.extract_all(img_paths)
        else:
            self.img_features = self.fe.extract_all(img_paths)

        for i in range(len(img_paths)):
            self.img_paths[img_count + i] = img_paths[i]

    def rank_top_k(self, query_img_path, k):
        query_feature = self.fe.extract(query_img_path)
        img_count = len(self.img_features)
        query_diff = []
        for img_id in range(img_count):
            img_feature = self.img_features[img_id]
            query_diff.append(img_feature - query_feature)
        dist = np.linalg.norm(query_diff, axis=1)
        ids = np.argsort(dist)[:min(len(dist), k)]
        dist = [dist[id] for id in ids]
        return ids, dist

    def get_img(self, img_ids):
        return [self.img_paths[img_id] for img_id in img_ids]

    def query_top_k(self, query_img_path, k):
        ids, dist = self.rank_top_k(query_img_path, k)
        result = []
        for i in range(len(ids)):
            img_id = ids[i]
            rank = dist[i]
            img_path = self.img_paths[img_id]
            result.append({
                'img_id': int(img_id),
                'rank': float(rank),
                'path': img_path
            })
        return result

    def do_default_indexing(self):
        for file in os.listdir(DATA_DIR_PATH):
            file_path = os.path.join(DATA_DIR_PATH, file)
            if os.path.isfile(file_path):
                print('processing file: ', file)
                self.index_image(file_path)

    def img_count(self):
        return len(self.img_features)

    def test_run(self):
        if self.img_count() == 0:
            self.index_images([os.path.join(DATA_DIR_PATH, 'Pokemon1.jpg'), os.path.join(DATA_DIR_PATH, 'Pokemon2.jpg')])
        print(self.rank_top_k(os.path.join(DATA_DIR_PATH, 'Pokemon1.jpg'), k=30))


def main():
    search_engine = VGG16ImageSearchEngine()
    search_engine.test_run()

if __name__ == '__main__':
    main()
