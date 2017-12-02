from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers.core import RepeatVector
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras.preprocessing import sequence
import sys
import collections
import nltk
import numpy as np
import os
import urllib.request
import zipfile

INPUT_FILE = "../../../data/umich-sentiment-train.txt"
VOCAB_SIZE = 5000
EMBED_SIZE = 100
BATCH_SIZE = 64
NUM_EPOCHS = 10

LATENT_SIZE = 512
MAX_SEQ_LEN = 50

VERY_LARGE_DATA_DIR = './very_large_data'
NEWS_DATA_DIR = './data/news'
GLOVE_MODEL = VERY_LARGE_DATA_DIR + "/glove.6B." + str(EMBED_SIZE) + "d.txt"
TRAINING_FILE_PATH = os.path.join(NEWS_DATA_DIR, "reuters-21528-text.tsv")
MODEL_DIR = './models'


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
        zip_ref.extractall(VERY_LARGE_DATA_DIR + '')
        zip_ref.close()


def lookup_word2id(word2id, word):
    try:
        return word2id[word]
    except KeyError:
        return word2id['UNK']


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


def sentence_generator(X, embeddings, batch_size):
    while True:
        # loop once per epoch
        num_recs = X.shape[0]
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            sids = indices[bid * batch_size: (bid + 1) * batch_size]
            Xbatch = embeddings[X[sids, :]]
            yield Xbatch, Xbatch


def compute_cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x, 2) * np.linalg.norm(y, 2))


# parsing sentences and building vocabulary
word_freqs = collections.Counter()
ftext = open(TRAINING_FILE_PATH, "rt")
sents = []
for line in ftext:
    docid, text = line.strip().split("\t")
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            word = word.lower()
            word_freqs[word] += 1
        sents.append(sent)
ftext.close()


word2id = dict()
word2id["PAD"] = 0
word2id["UNK"] = 1
for v, (k, _) in enumerate(word_freqs.most_common(VOCAB_SIZE - 2)):
    word2id[k] = v + 2
id2word = {v: k for k, v in word2id.items()}

print("vocabulary sizes:", len(word2id), len(id2word))

np.save(os.path.join(MODEL_DIR, 'sent-autoencoder-word2id.npy'), word2id)
np.save(os.path.join(MODEL_DIR, 'sent-autoencoder-id2word.npy'), id2word)

sent_wids = [[lookup_word2id(word2id, w) for w in s.split()] for s in sents]
sent_wids = sequence.pad_sequences(sent_wids, MAX_SEQ_LEN)

# load glove vectors into weight matrix
embeddings = load_glove_vectors(word2id, EMBED_SIZE)
print(embeddings.shape)

# split sentences into training and test
train_size = 0.7
Xtrain, Xtest = train_test_split(sent_wids, train_size=train_size)
print("number of sentences: ", len(sent_wids))
print(Xtrain.shape, Xtest.shape)

# define training and test generators
train_gen = sentence_generator(Xtrain, embeddings, BATCH_SIZE)
test_gen = sentence_generator(Xtest, embeddings, BATCH_SIZE)

# define autoencoder network
inputs = Input(shape=(MAX_SEQ_LEN, EMBED_SIZE), name="input")
encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum",
                        name="encoder_lstm")(inputs)
decoded = RepeatVector(MAX_SEQ_LEN, name="repeater")(encoded)
decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True),
                        merge_mode="sum",
                        name="decoder_lstm")(decoded)

auto_encoder = Model(inputs, decoded)

auto_encoder.compile(optimizer="sgd", loss="mse")


# train
num_train_steps = len(Xtrain) // BATCH_SIZE
num_test_steps = len(Xtest) // BATCH_SIZE
checkpoint = ModelCheckpoint(filepath=os.path.join(MODEL_DIR, "sent-autoencoder.h5"),
                             save_best_only=True)
history = auto_encoder.fit_generator(train_gen,
                                     steps_per_epoch=num_train_steps,
                                     epochs=NUM_EPOCHS,
                                     validation_data=test_gen,
                                     validation_steps=num_test_steps,
                                     callbacks=[checkpoint])

auto_encoder.save_weights(filepath=os.path.join(MODEL_DIR, 'sent-autoencoder.h5'))



