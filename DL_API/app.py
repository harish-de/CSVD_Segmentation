from flask import Flask, render_template, request, flash, redirect
import nibabel as nib
import dicom2nifti as d2n

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def g3t_data():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
        print(file)
        #img_arr = nib.load(file).get_fdata()
        return render_template('result.html')
        # if file.endswith('.dicom'):
        #     # data = d2n.dicom_series_to_nifti(file, file, reorient_nifti=True)
        #     return ('The data is a dicom file')
        # elif file.endswith('.nii'):
        #     # data = nib.load(file)
        #     return ('The data is a NifTi file')


# def upload_file():
#     if request.method == 'POST':
#         # check if the post request has the file part
#         if 'file' not in request.files:
#             flash('No file part')
#             return redirect(request.url)
#         file = request.files['file']
#         # if user does not select file, browser also
#         # submit an empty part without filename
#         if file.filename == '':
#             flash('No selected file')
#             return redirect(request.url)
#         if file and allowed_file(file.filename):
#             print(allowed_file)


if __name__ == '__main__':
    app.run(debug=True)
