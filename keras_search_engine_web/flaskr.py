from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from keras_search_engine_web.glove_doc_search_engine import GloveDocSearchEngine
from keras_search_engine_web.vgg16_img_search_engine import VGG16ImageSearchEngine

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


glove_doc_search_engine = GloveDocSearchEngine()
vgg16_image_search_engine = VGG16ImageSearchEngine()

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/search_text_glove', methods=['POST', 'GET'])
def search_text_glove():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            search_result = glove_doc_search_engine.rank_top_k(sent)
            return render_template('search_text_glove.html', sentence=sent,
                                   search_result=search_result)
    return render_template('search_text_glove.html')


@app.route('/search_text', methods=['POST', 'GET'])
def measure_sentiment():
    if request.method == 'POST':
        if not request.json or 'sentence' not in request.json or 'model' not in request.json \
                or 'limit' not in request.json:
            abort(400)
        sentence = request.json['sentence']
        model = request.json['model']
        limit = request.json['limit']
    else:
        sentence = request.args.get('sentence')
        model = request.args.get('model')
        limit = request.args.get('limit')

    docs = []
    if model == 'glove':
        docs = glove_doc_search_engine.query_top_k(sentence, k=limit)
    return jsonify({
        'sentence': sentence,
        'result': docs,
        'model': model
    })


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


def main():
    glove_doc_search_engine.do_default_indexing()
    glove_doc_search_engine.test_run()

    vgg16_image_search_engine.do_default_indexing()
    vgg16_image_search_engine.test_run()

    app.run(debug=True)

if __name__ == '__main__':
    main()
