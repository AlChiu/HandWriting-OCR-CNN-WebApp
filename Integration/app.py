from flask import Flask, render_template, request
from werkzeug import secure_filename
import os
import detector
import classifier

app=Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
	if request.method == 'POST':
		f = request.files['file']
		f.save(secure_filename(f.filename))  #Save image at the server
		detector.detector(f.filename)
		words = classifier.classify()
		os.remove(f.filename)
		return render_template("show_entries.html", data=words)

@app.route('/about', methods =['GET'])
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug = True)
