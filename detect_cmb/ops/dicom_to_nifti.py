import os
import re
import dicom2nifti as d2n

'''
To convert DCIM to .Nii format
'''


class prepareNiftiFromDicom():

    def createNifti(self, dicom_dir, nifti_dir):
        '''
        :param dicom_dir: path containing input Dicom files
        :param nifti_dir: path where Nifti file has to be saved
        '''

        for main_folder in os.listdir(dicom_dir):
            output = nifti_dir
            if ((main_folder.endswith(".zip") or (main_folder.endswith(".csv"))) == False):
                os.chdir(nifti_dir)
                os.mkdir(main_folder)
                output_temp = os.path.join(output, main_folder)
                main_target = os.path.join(dicom_dir, main_folder)
                os.chdir(main_target)

                for folder in os.listdir('.'):
                    target = os.path.join(main_target, folder)
                    os.chdir(target)

                    for subfolder in os.listdir('.'):
                        if (subfolder.endswith(".nii") == False):
                            newtarget = os.path.join(target, subfolder)
                            os.chdir(newtarget)

                            for nextsubfolder in os.listdir('.'):
                                filename_1 = folder[-4:]
                                filename = nextsubfolder
                                filename = filename[:-11]
                                filename = re.sub('[-]', '', filename)
                                filename = filename_1 + '_' + filename + '.nii'

                                output = os.path.join(output_temp, filename)
                                final_target = os.path.join(newtarget, nextsubfolder)
                                try:
                                    d2n.dicom_series_to_nifti(final_target, output, reorient_nifti=True)
                                except:
                                    print(filename + 'is not created due to conversion errors')
