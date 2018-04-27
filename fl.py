from flask import Flask, request, render_template
import operations
from flask_cors import CORS
import json
from werkzeug.datastructures import ImmutableMultiDict

app = Flask(__name__, template_folder='static')
CORS(app)

@app.route("/init", methods=['GET','POST'])
def train():
	if request.method == 'POST':
		dt = {
			'dataset_dir': request.form['dataset_dir']
		}
	else:
		dt = {
			'dataset_dir': request.args.get('dir')
		}

	return operations.train(dt)

@app.route("/test", methods=["POST"])
def upload():
	#print((request.form.getlist(['files'])))
	uploaded_files = request.files['file']
	uploaded_files.save(uploaded_files.filename)
	return json.dumps(operations.inference(uploaded_files.filename))



if __name__ == "__main__":
	app.run(host='0.0.0.0', port=8080)