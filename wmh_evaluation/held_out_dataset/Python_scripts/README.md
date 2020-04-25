
### preprocessing.py
* specify respective paths against GE3T_DIR, SINGAPORE_DIR, and UTRECHT_DIR
* create directories as 'dir_input', 'dir_output', 'out_dir_input_slices', 'out_dir_groudtruth_slices'
* takes randomly 17/20 subjects from each hospital for preprocessing
* outputs 'data_tensor.sav' containing preprocessed 2D axial slices and 
* 'test_imgs.txt' - list of test subjects unseen during training

### train_network.py
* create directory checkpoints to save each epoch model before executing this script
* outputs ''plot_loss.txt' containing list of epoch index vs training & validation loss summary
