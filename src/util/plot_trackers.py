import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_original = pd.read_csv('./data/Tracker.csv')
df_aligned = pd.read_csv('./data/Tracker_aligned.csv')

# Plot original tracker trajectory
plt.figure()
plt.plot(df_original['x'], df_original['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tracker Original (x vs y)')
plt.savefig('./figures/tracker_original.png')
plt.show()

# Plot aligned tracker trajectory
plt.figure()
plt.plot(df_aligned['x'], df_aligned['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Tracker Aligned (x vs y)')
plt.savefig('./figures/tracker_aligned.png')
plt.show()
