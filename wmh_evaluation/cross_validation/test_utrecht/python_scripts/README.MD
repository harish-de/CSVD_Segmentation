
### preproc_sing_ge3t.py 

* specify respective paths against GE3T_DIR, SINGAPORE_DIR
* provide augmented files of Ge3t against 'GE3T_AUG_DIR'
* provide augmented files of singapore against 'SINGAPORE_AUG_DIR'
* create directories as 'dir_input', 'dir_output', 'out_dir_input_slices', 'out_dir_groudtruth_slices'
* outputs 'data_sing_ge3t.sav' containing preprocessed 2D axial slices  


### train_sing_ge3t.py
* create directory 'checkpoints_sing_ge3t' to save each epoch model before executing this script
* outputs 'plot_loss_sing_ge3t.txt' containing list of epoch index vs training & validation loss summary

### test_utrecht_cv.py

* Provide the path for utrecht dataset in 'HOME'
* specify suitable checkpoint_index against path (as in path = 'checkpoints\\checkpoint_30.pth.tar')
* outputs 'evaluation_utrecht.txt' containing metric scores of all Utrecht subjects
