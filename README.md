# keras-search-engine

A simple document and image search engine implemented in keras

# Features

* Document level encoding using GloVe Word Embedding
* Support Full Text Search using GloVe Embedding
* Support Document Indexing via web api
* Support Image Search
* Support Image Indexing via web api (WIP)
* Document auto-encoding using LSTM encoder-decoder with GloVe Embedding

# Usage

## Running Web Api Server

Goto keras_search_engine_web directory and run the following command:

```bash
python flaskr.py
```

Now navigate your browser to http://localhost:5000 and you can try out
 
* the story book search (by which user enters search query on the story books that are loaded from keras_search_engine_train/data/texts)
* the image search (by which user uploads an image to search among images stored in the keras_search_engine_train/data/images)

## Invoke web api to index text document

With the web api server running, you can index a new document by calling the following web api:

```html
http://localhost:5000/index_text
```

For example the following is the curl command to call the web api to index some documents:

```bash
curl -H 'Content-Type application/json' -X POST -d '{"doc":"Whether you think that you can, or that you can't, you are usually right."}' http://localhost:5000/index_text
curl -H 'Content-Type application/json' -X POST -d '{"doc":"Try to learn something about everything and everything about something."}' http://localhost:5000/index_text
curl -H 'Content-Type application/json' -X POST -d '{"doc":"You can avoid reality, but you cannot avoid the consequences of avoiding reality."}' http://localhost:5000/index_text
curl -H 'Content-Type application/json' -X POST -d '{"doc":"A mathematician is a device for turning coffee into theorems."}' http://localhost:5000/index_text
curl -H 'Content-Type application/json' -X POST -d '{"doc":"In theory, there is no difference between theory and practice. But in practice, there is."}' http://localhost:5000/index_text
curl -H 'Content-Type application/json' -X POST -d '{"doc":"I find that the harder I work, the more luck I seem to have."}' http://localhost:5000/index_text
```

To query using the web api, you can call the following web api:

```bash
curl -H 'Content-Type application/json' -X POST -d '{"query":"mathematician and coffee", "limit": 3, "model": "glove"}' http://localhost:5000/search_text
```

## Invoke web api to index text document

With the web api server running, you can index a new image by calling the following web api via POST request:

```html
http://localhost:5000/index_image
```

You can query similar images by calling the following web api POST request:

```html
http://localhost:5000/search_image/10
```

where 10 is the limit on the number of images returned

For example the following python code the web api to index an image:

```python 

```

To query using the web api, you can run the following python:

```python

```


