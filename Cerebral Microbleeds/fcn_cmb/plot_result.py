import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('discrimination_stage2_plot_loss.csv', header=0)

print(df.columns.tolist())

epoch_count = df['Current_epoch']
train_loss = df['Train_loss']
valid_loss = df['Validation_loss']

# plt.plot(epoch_count,[train_loss,valid_loss])

df.plot(x="Current_epoch", y=["Train_loss", "Validation_loss"])
plt.show()