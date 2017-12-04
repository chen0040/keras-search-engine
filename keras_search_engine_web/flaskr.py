from flask import Flask, request, send_from_directory, send_file, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from keras_search_engine_web.glove_doc_search_engine import GloveDocSearchEngine
from keras_search_engine_web.vgg16_img_search_engine import VGG16ImageSearchEngine
from keras_search_engine_web.glove_sent_encoder_search_engine import GloveDocEncoderSearchEngine
import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable

app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

glove_story_search_engine = GloveDocSearchEngine()

glove_doc_search_engine = GloveDocSearchEngine()
vgg16_image_search_engine = VGG16ImageSearchEngine()
glove_doc_encoder_search_engine = GloveDocEncoderSearchEngine()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def store_uploaded_image(action):
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for(action,
                                filename=filename))


@app.route('/get_image/<filename>')
def get_image(filename):
    return send_file(filename, mimetype='image/jpg')


@app.route('/search_vgg16_image', methods=['GET', 'POST'])
def search_vgg16_image():
    if request.method == 'POST':
        return store_uploaded_image('search_vgg16_image_result')
    return render_template('search_vgg16_image.html')


@app.route('/search_vgg16_image_result/<filename>')
def search_vgg16_image_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    top5 = vgg16_image_search_engine.query_top_k(filepath, k=5)
    return render_template('search_vgg16_image_result.html', filename=filename,
                           search_result=top5)


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


@app.route('/index_image', methods=['POST'])
def index_image():
    # check if the post request has the file part
    if 'file' not in request.files:
        return make_response(jsonify({'error': 'Uploaded file not found'}), 404)
    file = request.files['file']
    if file.filename == '':
        return make_response(jsonify({'error': 'Uploaded filename is blank'}), 404)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img_feature = vgg16_image_search_engine.index_images(file_path)
        return jsonify({
            'doc_feature': img_feature
        })
    else:
        return make_response(jsonify({'error': 'Invalid file format'}), 404)


@app.route('/search_image/<limit>', methods=['POST'])
def search_image(limit):
    if 'file' not in request.files:
        return make_response(jsonify({'error': 'Uploaded file not found'}), 404)
    file = request.files['file']
    if file.filename == '':
        return make_response(jsonify({'error': 'Uploaded filename is blank'}), 404)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        images = vgg16_image_search_engine.query_top_k(file_path, k=limit)
        return jsonify({
            'query': file_path,
            'result': images
        })
    else:
        return make_response(jsonify({'error': 'Invalid file format'}), 404)


@app.route('/index_text', methods=['POST', 'GET'])
def index_text():
    if request.method == 'POST':
        if not request.json or 'doc' not in request.json:
            abort(400)
        doc = request.json['doc']
    else:
        doc = request.args.get('doc')

    doc_feature = glove_doc_search_engine.index_document(doc)
    doc_encoder_feature = glove_doc_encoder_search_engine.index_document(doc)
    return jsonify({
        'glove_doc_feature': doc_feature,
        'encoder_doc_feature': doc_encoder_feature
    })


@app.route('/doc_count', methods=['GET'])
def doc_count():
    return jsonify({
        'glove_doc_count': glove_doc_search_engine.doc_count(),
        'encoder_doc_count': glove_doc_encoder_search_engine.doc_count()
    })


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
    elif model == 'doc-encoder':
        docs = glove_doc_encoder_search_engine.query_top_k(query, k=limit)
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
    glove_doc_encoder_search_engine.test_run()

    vgg16_image_search_engine.do_default_indexing()
    vgg16_image_search_engine.test_run()

    app.run(debug=True)


if __name__ == '__main__':
    main()
