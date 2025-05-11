import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  

df_tracker = pd.read_csv('./data/Tracker.csv')
df_accel = pd.read_csv('./data/Accelerometer.csv')

# Merge the two dataframes on the 'Time' column
# They use two different scales, where acc 

# Determine the common end time (shorter of the two)
end_time = df_tracker['t'].max()

# Trim accelerometer data to the same duration
df_accel_trimmed = df_accel[df_accel['seconds_elapsed'] <= end_time].reset_index(drop=True)

# Perform linear interpolation of tracker values at accel timestamps
for col in ['x', 'y', 'vx', 'vy']:
    df_accel_trimmed[col] = np.interp(
        df_accel_trimmed['seconds_elapsed'],
        df_tracker['t'],
        df_tracker[col]
    )

# Build aligned DataFrame
aligned_df = df_accel_trimmed[['time', 'seconds_elapsed', 'x', 'y', 'vx', 'vy']]

# Save to CSV
aligned_df.to_csv('./data/Tracker_aligned.csv', index=False)

# Display results
print(f"Common end time: {end_time:.6f} seconds")
print(f"Aligned rows: {len(aligned_df)}")
print(f"Last seconds_elapsed: {aligned_df['seconds_elapsed'].iloc[-1]:.6f} seconds\n")

# Show the first 10 rows for the aligned dataframe
aligned_df.head(10)