from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
import numpy as np
import pandas as pd
import os
from pathlib import Path
import pathlib
import sys
# sys.path.insert(1, 'D:\src\email-2-faq\src')

temp = pathlib.Path(__file__).parent.resolve()
print("temp is :", temp)
path1 = os.path.dirname(temp)
print(os.path.dirname(temp))

sys.path.insert(1, path1)

import fgen

path2 = os.path.join(path1, 'web', 'static')
print("path2 is ", path2)

Path(path2).mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER = path2
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app._static_folder = 'static'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = 'input.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filepath, file=sys.stderr) 
            file.save(filepath)
        
        # The predicted output from the model
        pred = fgen.generate_faq_fgen(filepath)
        # print(pred)
        
        # Uncomment the pred below later - sample prediction
        # pred = {'valid_queries': ["what are you upto", "is everything ok", "what are different types of laptops available", "what are specifications of each type of laptop"], 'query_clusters': [['is everything ok', 'what will be good specs of the gaming laptop'], ['what are different types of laptops available', 'what is warranty period of laptops']], 'valid_faq': ['what is best gaming laptop?', 'what is the best laptop to buy?']}
        
        
        return render_template('result.html', pred = pred)
    
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=5000, debug=True)