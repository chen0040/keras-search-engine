from flask import Flask, request, send_from_directory, redirect, render_template, flash, url_for, jsonify, \
    make_response, abort
from keras_search_engine_web.wordvec_glove_feature_extractor import WordVecGloveFeatureExtractor

app = Flask(__name__)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


fe_glove_c = WordVecGloveFeatureExtractor()
fe_glove_c.test_run('i like the Da Vinci Code a lot.')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/about')
def about():
    return 'About Us'


@app.route('/ffn_glove', methods=['POST', 'GET'])
def ffn_glove():
    if request.method == 'POST':
        if 'sentence' not in request.form:
            flash('No sentence post')
            redirect(request.url)
        elif request.form['sentence'] == '':
            flash('No sentence')
            redirect(request.url)
        else:
            sent = request.form['sentence']
            sentiments = fe_glove_c.predict(sent)
            return render_template('ffn_glove_result.html', sentence=sent,
                                   sentiments=sentiments)
    return render_template('ffn_glove.html')


@app.route('/measure_sentiments', methods=['POST', 'GET'])
def measure_sentiment():
    if request.method == 'POST':
        if not request.json or 'sentence' not in request.json or 'network' not in request.json:
            abort(400)
        sentence = request.json['sentence']
        network = request.json['network']
    else:
        sentence = request.args.get('sentence')
        network = request.args.get('network')

    sentiments = []
    if network == 'cnn':
        sentiments = fe_glove_c.predict(sentence)
    elif network == 'ffn_glove':
        sentiments = fe_glove_c.predict(sentence)
    return jsonify({
        'sentence': sentence,
        'pos_sentiment': float(str(sentiments[0])),
        'neg_sentiment': float(str(sentiments[1])),
        'network': network
    })


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    app.run(debug=True)
