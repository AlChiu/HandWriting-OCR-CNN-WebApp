from flask import Flask, render_template, request
from werkzeug import secure_filename
import cv2
app=Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))  #Save image at the server
        
        image = cv2.imread(f.filename)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('gray_image/gray.jpeg', gray_image)   #save grayscale image at the server 
        
        return 'Image sucessfully converted to Grayscale'
        
if __name__ == "__main__":
    app.run(debug = True)