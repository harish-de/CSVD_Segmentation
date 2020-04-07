from flask import Flask, request, render_template, jsonify, flash, redirect, url_for, sessions, send_from_directory
from werkzeug.utils import secure_filename
import os
from flask_session import Session
import test
import natsort

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static')
ALLOWED_EXTENSIONS = {'nii','gz'}


app = Flask(__name__)
sess = Session()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and request.form['lesion'] == 'WMH':
        uploaded_files = request.files.getlist("image")
        filenames = []
        for file in uploaded_files:
            filename = secure_filename(file.filename)
            filenames.append(filename)

            if filenames == ['']:
                return render_template('fileerror.html')

            if filename.find("FLAIR") == 0 or filename.find("flair") == 0:
                if file and allowed_file(file.filename):
                    # filename = secure_filename(file.filename)
                    filename = 'image_flair.nii.gz'
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                else:
                    return render_template('fileerror.html')

            elif filename.find("T1") == 0 or filename.find("t1") == 0:

                if file and allowed_file(file.filename):
                    # filename = secure_filename(file.filename)
                    filename = 'image_t1.nii.gz'
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                else:
                    return render_template('fileerror.html')
            else:
                return render_template('fileerror.html')

        test.main()
        os.remove('static/image_flair.nii.gz')
        os.remove('static/image_t1.nii.gz')
        return render_template('viewer.html')
    elif request.method == 'POST' and request.form['lesion'] == 'CMB':
        return render_template('cmb.html')
    else:
        return render_template('fileerror.html')

    #For showing slices
    # hists = (os.listdir('49'))
    # hists = natsort.natsorted(hists, reverse=False)
    # hists = [file for file in hists]
    # return render_template('show_slices.html', hists=hists)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/')
def home():
    return render_template('home.html')


if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'
    sess.init_app(app)
    app.debug = False
    app.run()
