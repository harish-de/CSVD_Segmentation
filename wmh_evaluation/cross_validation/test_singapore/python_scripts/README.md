### preproc_utr_ge3t.py 

* specify respective paths against GE3T_DIR and UTRECHT_DIR
* provide augmented files of Ge3t against 'GE3T_AUG_DIR'
* provide augmented files of utrecht against 'UTRECHT_AUG_DIR'
* create directories as 'dir_input', 'dir_output', 'out_dir_input_slices', 'out_dir_groudtruth_slices'
* outputs 'data_utr_ge3t.sav' containing preprocessed 2D axial slices  

### train_utr_ge3t.py
* create directory 'checkpoints_utr_ge3t' to save each epoch model before executing this script
* outputs 'plot_loss_utr_ge3t.txt' containing list of epoch index vs training & validation loss summary

### test_singapore_cv.py

* Provide the path for singapore dataset in 'HOME'
* specify suitable checkpoint_index against path (as in path = 'checkpoints\\checkpoint_30.pth.tar')
* outputs 'evaluation_singapore.txt' containing metric scores of all Amsterdam_GE3T subjects

