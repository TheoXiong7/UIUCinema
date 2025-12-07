from flask import Flask, render_template, request, jsonify
from models import create_search_engine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'stupid-key'
app.config['TEMPLATES_AUTO_RELOAD'] = True

search_engine = None
model_status = {'loaded': False, 'error': None}


def load_model():
    global search_engine, model_status

    try:
        print('Initializing...')

        search_engine = create_search_engine(
            data_dir='data',
            models_dir='models'
        )

        model_status['loaded'] = True
        print('model loaded.')

    except Exception as e:
        model_status['error'] = str(e)
        print(f"\nERROR loading model: {e}\n")
        import traceback
        traceback.print_exc()


# Load model on startup
load_model()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['GET', 'POST'])
def search():
    query = request.args.get('q', '') or request.form.get('query', '')

    if not query:
        return render_template('results.html', query='', results=[],
                             message='Please enter a search query')

    if not model_status['loaded']:
        error_msg = f"Search engine not ready. Error: {model_status.get('error', 'Unknown error')}"
        return render_template('results.html', query=query, results=[],
                             message=error_msg)

    try:
        results = search_engine.search(query, top_k=10)

        if not results:
            return render_template('results.html', query=query, results=[],
                                 message='No movies found matching your query.')

        return render_template('results.html', query=query, results=results)

    except Exception as e:
        print(f"Search error: {e}")
        return render_template('results.html', query=query, results=[],
                             message=f'Search error: {str(e)}')


@app.route('/api/search')
def api_search():
    query = request.args.get('q', '')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    if not model_status['loaded']:
        return jsonify({
            'error': 'Search engine not ready',
            'details': model_status.get('error', 'Unknown error')
        }), 503

    try:
        results = search_engine.search(query, top_k=10)

        return jsonify({
            'query': query,
            'count': len(results),
            'results': results,
            'model': 'two-stage-hybrid',
            'alpha': search_engine.alpha
        })

    except Exception as e:
        return jsonify({
            'error': 'Search failed',
            'details': str(e)
        }), 500


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/api/status')
def api_status():
    """API endpoint to check model status."""
    return jsonify({
        'model_loaded': model_status['loaded'],
        'error': model_status.get('error'),
        'search_available': model_status['loaded'] and search_engine is not None,
        'model_config': {
            'alpha': search_engine.alpha if search_engine else None,
            'stage1_k': search_engine.stage1_k if search_engine else None
        } if model_status['loaded'] else None
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
