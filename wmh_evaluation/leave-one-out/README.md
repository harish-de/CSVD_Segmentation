
### preprocessing_leaveoneout.py
* Create 4 directories namely 'dir_input'  'dir_output'  'out_dir_input_slices' 'out_dir_groudtruth_slices' for storing the preprocessed images and slices respectively

### leave-one-out.py

* Running this file leaves one image for testing and trains all other images for all   60 possible combinations

* Checkpoints folder is created for every combination of leave one out and each folder has checkpoints at each epoch.

* This file returns a 'plot_loss_index.txt' containing list of epoch index vs training & validation loss summary,for every combinations of leave one out

* This file returns  a 'evaluation_leavoneout.txt' which cantains metric scores for each subject and also the average score for scores of all subjects
