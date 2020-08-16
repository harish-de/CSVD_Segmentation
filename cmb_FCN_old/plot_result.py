import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

df = pd.read_csv('plot_loss.txt', header=0)

print(df.columns.tolist())

epoch_count = df['Current_epoch']
train_loss = df['Train_loss']
valid_loss = df['Validation_loss']

# plt.plot(epoch_count,[train_loss,valid_loss])

ax = df.plot(x="Current_epoch", y=["Train_loss", "Validation_loss"])
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.xaxis.set_ticks(np.arange(85, 100, 1))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))

plt.show()
