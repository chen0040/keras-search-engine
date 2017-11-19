from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.optimizers import SGD
from PIL import Image
from keras.preprocessing.image import img_to_array
import numpy as np


class VGG16FeatureExtractor:
    model = None

    def __init__(self):
        self.model = VGG16(include_top=True, weights='imagenet')
        self.model.compile(optimizer=SGD(), loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, filename):
        img = Image.open(filename)
        img = img.resize((224, 224), Image.ANTIALIAS)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        output = decode_predictions(self.model.predict(input), top=3)
        return output[0]

    def extract(self, filename):
        img = Image.open(filename)
        img = img.resize((224, 224), Image.ANTIALIAS)
        input = img_to_array(img)
        input = np.expand_dims(input, axis=0)
        input = preprocess_input(input)
        feature = self.model.predict(input)[0]
        return feature / np.linalg.norm(feature)

    def extract_all(self, filenames):
        result = []
        for filename in filenames:
            result.append(self.extract(filename))
        return result

    def run_test(self):
        print(self.extract('../keras_search_engine_train/data/images/Pokemon1.jpg'))

if __name__ == '__main__':
    classifier = VGG16FeatureExtractor()
    classifier.run_test()
