
### test_imgs.txt
* Contains list of test subject location 
* 3 out of 20 subjects from each hospital were chosen randomly as unseen test cases 

### plot_epochloss_heldout.csv
* network is trained for 100 epochs
* contains training & validation loss against current epoch index

### Loss_Heldout_testing.png
* line graph plotted for epoch vs training & validation loss

### evaluation_heldout.txt
* gives the score of DSC,AVD, Recall and F1 for each test subject
* finally the average scores is computed for all 9 test subjects

**Avg. DSC** = 0.7863575362336371

**Avg. AVD** = 23.187817493329046

**Avg. Recall** = 0.9263190292952197

**Avg. F1** = 0.724768491392512

DSC - Dice Similarity Coefficient
AVD - Average Volume difference
