from flask import *
import nibabel as nib
import numpy as np
import os
import werkzeug
from werkzeug.utils import secure_filename

app = Flask(__name__)


@app.route('/')
def upload():
    return render_template("index1.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        return render_template('success.html', name=f.filename)


if __name__ == '__main__':
    app.run(debug=True)
