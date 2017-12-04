import json
import urllib
import urllib.request


class SearchEngineClient(object):
    baseUrl = None

    def __init__(self):
        self.baseUrl = 'http://localhost:5000'

    def index_text(self, text):
        new_request = {"doc": text}
        params = json.dumps(new_request).encode('utf8')
        req = urllib.request.Request(self.baseUrl + '/index_text', data=params,
                                     headers={'content-type': 'application/json'})
        web_url = urllib.request.urlopen(req)
        data = web_url.read()
        encoding = web_url.info().get_content_charset('utf-8')
        response = data.decode(encoding)
        result = json.loads(response)
        return result

    def search_text(self, query, limit, model):
        new_request = {"query": query, "limit": limit, "model": model}
        params = json.dumps(new_request).encode('utf8')
        req = urllib.request.Request(self.baseUrl + '/search_text', data=params,
                                     headers={'content-type': 'application/json'})
        web_url = urllib.request.urlopen(req)
        data = web_url.read()
        encoding = web_url.info().get_content_charset('utf-8')
        response = data.decode(encoding)
        result = json.loads(response)
        print(response)
        return result

    def doc_count(self):
        web_url = urllib.request.urlopen(self.baseUrl + '/doc_count')
        data = web_url.read()
        encoding = web_url.info().get_content_charset('utf-8')
        result = json.loads(data.decode(encoding))
        return int(result['glove_doc_count'])


def main():
    client = SearchEngineClient()
    doc_count = client.doc_count()
    if doc_count < 4:
        client.index_text('Whether you think that you can, or that you can.')
        client.index_text('Try to learn something about everything and everything about something.')
        client.index_text('You can avoid reality, but you cannot avoid the consequences of avoiding reality.')
        client.index_text('A mathematician is a device for turning coffee into theorems.')
        client.index_text('In theory, there is no difference between theory and practice. But in practice, there is.')
        client.index_text('I find that the harder I work, the more luck I seem to have.')
    client.search_text(query='mathematician and coffee', limit=3, model='glove')
    # client.search_text(query='mathematician and coffee', limit=3, model='doc-encoder')


if __name__ == '__main__':
    main()
