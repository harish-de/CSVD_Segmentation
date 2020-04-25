
### augmentation.py

* Provide 3 paths:'utrecht_dir', 'singapore_dir', 'ge3t_dir'  
* Make 3 directories namely: 'utrecht', 'singapore', 'ge3t', to store the augmented files

### preproc_utr_sin.py 

* specify respective paths against GE3T_DIR, SINGAPORE_DIR, and UTRECHT_DIR
* create directories as 'dir_input', 'dir_output', 'out_dir_input_slices', 'out_dir_groudtruth_slices'
* outputs 'data_utr_sing.sav' containing preprocessed 2D axial slices  

### train_utr_sing.py
* create directory 'checkpoints_utr_sin' to save each epoch model before executing this script
* outputs 'plot_loss_utr_sin.txt' containing list of epoch index vs training & validation loss summary

### test_ge3t_cv.py

* specify suitable checkpoint_index against path (as in path = 'checkpoints\\checkpoint_30.pth.tar')
* outputs 'evaluation_ge3t.txt' containing metric scores of all Amsterdam_GE3T subjects
