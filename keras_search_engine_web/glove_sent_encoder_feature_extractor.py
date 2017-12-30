from keras.models import Model
from keras.preprocessing import sequence
from keras.layers import Input, LSTM, Bidirectional, RepeatVector
import numpy as np
import os
import sys
import zipfile
import urllib.request

LATENT_SIZE = 512
MAX_SEQ_LEN = 50
EMBED_SIZE = 100
VERY_LARGE_DATA_DIR = '../keras_search_engine_train/very_large_data'
MODEL_DIR = '../keras_search_engine_train/models'
GLOVE_MODEL = VERY_LARGE_DATA_DIR + "/glove.6B." + str(EMBED_SIZE) + "d.txt"


def reporthook(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = read_so_far * 1e2 / total_size
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(total_size)), read_so_far, total_size)
        sys.stderr.write(s)
        if read_so_far >= total_size:  # near the end
            sys.stderr.write("\n")
    else:  # total size is unknown
        sys.stderr.write("read %d\n" % (read_so_far,))


def download_glove():
    if not os.path.exists(GLOVE_MODEL):
        if not os.path.exists(VERY_LARGE_DATA_DIR):
            os.makedirs(VERY_LARGE_DATA_DIR)

        glove_zip = VERY_LARGE_DATA_DIR + '/glove.6B.zip'

        if not os.path.exists(glove_zip):
            print('glove file does not exist, downloading from internet')
            urllib.request.urlretrieve(url='http://nlp.stanford.edu/data/glove.6B.zip', filename=glove_zip,
                                       reporthook=reporthook)

        print('unzipping glove file')
        zip_ref = zipfile.ZipFile(glove_zip, 'r')
        zip_ref.extractall(VERY_LARGE_DATA_DIR)
        zip_ref.close()


def load_glove_vectors(word2id, embed_size):
    download_glove()
    glove_file = os.path.join(VERY_LARGE_DATA_DIR, "glove.6B.{:d}d.txt".format(EMBED_SIZE))
    embedding = np.zeros((len(word2id), embed_size))
    fglove = open(glove_file, "rb")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        if embed_size == 0:
            embed_size = len(cols) - 1
        if word in word2id:
            vec = np.array([float(v) for v in cols[1:]])
            embedding[lookup_word2id(word2id, word)] = vec
    embedding[word2id["PAD"]] = np.zeros(shape=embed_size)
    embedding[word2id["UNK"]] = np.random.uniform(-1, 1, embed_size)
    return embedding


def lookup_word2id(word2id, word):
    if word in word2id:
        return word2id[word]
    return 1


class WordVecGloveDocFeatureExtractor(object):
    auto_encoder = None
    embedding = None
    word2id = None
    context = None

    def __init__(self):
        self.word2id = np.load(os.path.join(MODEL_DIR, 'sent-autoencoder-word2id.npy')).item()
        self.embedding = load_glove_vectors(self.word2id, EMBED_SIZE)

        # define auto-encoder network
        inputs = Input(shape=(MAX_SEQ_LEN, EMBED_SIZE), name="input")
        encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum",
                                name="encoder_lstm")(inputs)
        decoded = RepeatVector(MAX_SEQ_LEN, name="repeater")(encoded)
        decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True),
                                merge_mode="sum",
                                name="decoder_lstm")(decoded)

        self.auto_encoder = Model(inputs, decoded)

        self.auto_encoder.load_weights(filepath=os.path.join(MODEL_DIR, 'sent-autoencoder.h5'))
        self.auto_encoder.compile(optimizer="sgd", loss="mse")

    def extract(self, sentence):
        sent_wids = [[lookup_word2id(self.word2id, w) for w in sentence.split()]]
        sent_wids = sequence.pad_sequences(sent_wids, MAX_SEQ_LEN)
        X = self.embedding[sent_wids]
        return self.auto_encoder.predict(X).flatten()

    def extract_all(self, sentences):
        doc_count = len(sentences)

        result = []
        for k in range(doc_count):
            doc = sentences[k]
            result.append(self.extract(doc))
        return result

    def test_run(self):
        print(self.extract('i liked the Da Vinci Code a lot.'))


def main():
    app = WordVecGloveDocFeatureExtractor()
    app.test_run()


if __name__ == '__main__':
    main()
