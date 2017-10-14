from flask import Flask, render_template, request
from werkzeug import secure_filename
import cv2
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
        
        return render_template("show_entries.html", data=words)
        
if __name__ == "__main__":
    app.run(debug = True)