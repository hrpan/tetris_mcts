import pandas as pd
import matplotlib.pyplot as plt
import sys

file_path = sys.argv[1]

loss = pd.read_pickle(file_path)

l_plot = loss['loss'].plot(legend=True)
l_v_plot = loss['loss_value'].plot(legend=True)
l_p_plot = loss['loss_policy'].plot(legend=True)
l_val_plot = loss['loss_val'].plot(legend=True)
l_val_v_plot = loss['loss_val_value'].plot(legend=True)
l_val_p_plot = loss['loss_val_policy'].plot(legend=True)

plt.show()
