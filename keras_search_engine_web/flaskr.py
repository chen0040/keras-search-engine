from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from keras_search_engine_web.glove_doc_search_engine import GloveDocSearchEngine
from keras_search_engine_web.vgg16_img_search_engine import VGG16ImageSearchEngine

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


glove_story_search_engine = GloveDocSearchEngine()

glove_doc_search_engine = GloveDocSearchEngine()
vgg16_image_search_engine = VGG16ImageSearchEngine()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/search_story_glove', methods=['POST', 'GET'])
def search_story_glove():
    if request.method == 'POST':
        if 'query' not in request.form:
            flash('No query post')
            redirect(request.url)
        elif request.form['query'] == '':
            flash('No query')
            redirect(request.url)
        else:
            query = request.form['query']
            search_result = glove_story_search_engine.query_top_k(query, k=5)
            print(search_result)
            return render_template('search_story_glove.html', query=query,
                                   search_result=search_result)
    return render_template('search_story_glove.html')


@app.route('/search_text', methods=['POST', 'GET'])
def search_text():
    if request.method == 'POST':
        if not request.json or 'query' not in request.json or 'model' not in request.json \
                or 'limit' not in request.json:
            abort(400)
            query = request.json['query']
        model = request.json['model']
        limit = request.json['limit']
    else:
        query = request.args.get('query')
        model = request.args.get('model')
        limit = request.args.get('limit')

    docs = []
    if model == 'glove':
        docs = glove_doc_search_engine.query_top_k(query, k=limit)
    return jsonify({
        'query': query,
        'result': docs,
        'model': model
    })


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    glove_story_search_engine.do_default_indexing()
    glove_story_search_engine.test_run()
    glove_doc_search_engine.test_run()

    vgg16_image_search_engine.do_default_indexing()
    vgg16_image_search_engine.test_run()

    app.run(debug=True)

if __name__ == '__main__':
    main()
