import numpy as np
import nltk
import os
import sys
import zipfile
import urllib.request

MAX_SEQ_LENGTH = 2000
EMBED_SIZE = 100
VERY_LARGE_DATA_DIR = '../keras_search_engine_train/very_large_data'
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


class WordVecGloveFeatureExtractor(object):
    word2em = None
    context = None

    def __init__(self):
        self.word2em = load_glove()
        self.context = dict()
        self.context['maxlen'] = MAX_SEQ_LENGTH

    def extract(self, sentence):
        max_len = self.context['maxlen']
        tokens = [w.lower() for w in nltk.word_tokenize(sentence)]

        E = np.zeros(shape=(EMBED_SIZE, max_len))
        for j in range(0, min(len(tokens), max_len)):
            word = tokens[j]
            try:
                E[:, j] = self.word2em[word]
            except KeyError:
                pass
        return np.sum(E, axis=1)

    def extract_all(self, sentences):
        doc_count = len(sentences)
        max_len = self.context['maxlen']

        result = []
        for k in range(doc_count):
            tokens = [w.lower() for w in nltk.word_tokenize(sentences[k])]

            E = np.zeros(shape=(EMBED_SIZE, max_len))
            for j in range(0, len(tokens)):
                word = tokens[j]
                try:
                    E[:, j] = self.word2em[word]
                except KeyError:
                    pass
            result.append(np.sum(E, axis=1))
        return result

    def test_run(self):
        print(self.extract('i liked the Da Vinci Code a lot.'))
        print(self.extract_all(['Hello World', 'Hello Good Morning']))


def main():
    app = WordVecGloveFeatureExtractor()
    app.test_run()

if __name__ == '__main__':
    main()
