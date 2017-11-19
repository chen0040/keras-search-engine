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
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import urllib.request
import zipfile

INPUT_FILE = "../../../data/umich-sentiment-train.txt"
VOCAB_SIZE = 5000
EMBED_SIZE = 100
NUM_FILTERS = 256
NUM_WORDS = 3
BATCH_SIZE = 64
NUM_EPOCHS = 10

LATENT_SIZE = 512
SEQUENCE_LEN = 50

CACHE_DIR = './cache'
VERY_LARGE_DATA_DIR = './very_large_data'
NEWS_DATA_DIR = './data'
GLOVE_MODEL = VERY_LARGE_DATA_DIR + "/glove.6B." + str(EMBED_SIZE) + "d.txt"
TRAINING_FILE_PATH = os.path.join(NEWS_DATA_DIR, "reuters-21528-text.tsv")


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


def load_glove():
    download_glove()
    word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        word2em[word] = embeds
    file.close()
    return word2em


def lookup_word2id(word2id, word):
    try:
        return word2id[word]
    except KeyError:
        return word2id['UNK']


def load_glove_vectors(glove_file, word2id, embed_size):
    embedding = np.zeros((len(word2id), embed_size))
    fglove = open(glove_file, "rb")
    for line in fglove:
        cols = line.strip().split()
        word = cols[0]
        if embed_size == 0:
            embed_size = len(cols) - 1
        if word2id.has_key(word):
            vec = np.array([float(v) for v in cols[1:]])
        embedding[lookup_word2id(word)] = vec
    embedding[word2id["PAD"]] = np.zeros((embed_size))
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

sent_wids = [[lookup_word2id(w) for w in s.split()]
             for s in sents]
sent_wids = sequence.pad_sequences(sent_wids, SEQUENCE_LEN)

# load glove vectors into weight matrix
embeddings = load_glove_vectors(os.path.join(
    VERY_LARGE_DATA_DIR, "glove.6B.{:d}d.txt".format(EMBED_SIZE)), word2id, EMBED_SIZE)
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
inputs = Input(shape=(SEQUENCE_LEN, EMBED_SIZE), name="input")
encoded = Bidirectional(LSTM(LATENT_SIZE), merge_mode="sum",
                        name="encoder_lstm")(inputs)
decoded = RepeatVector(SEQUENCE_LEN, name="repeater")(encoded)
decoded = Bidirectional(LSTM(EMBED_SIZE, return_sequences=True),
                        merge_mode="sum",
                        name="decoder_lstm")(decoded)

autoencoder = Model(inputs, decoded)

autoencoder.compile(optimizer="sgd", loss="mse")

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# train
num_train_steps = len(Xtrain) // BATCH_SIZE
num_test_steps = len(Xtest) // BATCH_SIZE
checkpoint = ModelCheckpoint(filepath=os.path.join(CACHE_DIR, "sent-thoughts-autoencoder.h5"),
                             save_best_only=True)
history = autoencoder.fit_generator(train_gen,
                                    steps_per_epoch=num_train_steps,
                                    epochs=NUM_EPOCHS,
                                    validation_data=test_gen,
                                    validation_steps=num_test_steps,
                                    callbacks=[checkpoint])

# plot results
plt.plot(history.history["loss"], color="g", label="train")
plt.plot(history.history["val_loss"], color="b", label="validation")
plt.ylabel("loss (MSE)")
plt.xlabel("epochs")
plt.legend(loc="best")
plt.show()

# collect autoencoder predictions for test set
test_inputs, test_labels = test_gen.next()
preds = autoencoder.predict(test_inputs)

# extract encoder part from autoencoder
encoder = Model(autoencoder.input, autoencoder.get_layer("encoder_lstm").output)
# encoder.summary()

# compute difference between vector produced by original and autoencoded
k = 500
cosims = np.zeros(shape=k)
i = 0
for bid in range(num_test_steps):
    xtest, ytest = test_gen.next()
    ytest_ = autoencoder.predict(xtest)
    Xvec = encoder.predict(xtest)
    Yvec = encoder.predict(ytest_)
    for rid in range(Xvec.shape[0]):
        if i >= k:
            break
        cosims[i] = compute_cosine_similarity(Xvec[rid], Yvec[rid])
        if i <= 10:
            print(cosims[i])
        i += 1
    if i >= k:
        break

plt.hist(cosims, bins=10, normed=True)
plt.xlabel("cosine similarity")
plt.ylabel("frequency")
