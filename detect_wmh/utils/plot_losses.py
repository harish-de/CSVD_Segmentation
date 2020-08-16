import pandas as pd
import matplotlib.pyplot as plt

class plot_losses():
    def plot(self, csvFilename):

        df = pd.read_csv(csvFilename, header=0)

        epoch_count = df['Current_epoch']
        train_loss = df['Train_loss']
        valid_loss = df['Validation_loss']

        # plt.plot(epoch_count, [train_loss, valid_loss])

        df.plot(x="Current_epoch", y=["Train_loss", "Validation_loss"])
        plt.show()

plot_losses().plot('E:\\abhivanth\\detect_wmh\\plot_loss.txt')