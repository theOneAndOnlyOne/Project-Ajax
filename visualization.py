import matplotlib.pyplot as plt 
import pandas as pd 

# acceleration vs time

dataset = pd.read_csv('Josh Dataset\Walking\josh_back_fastwalking.csv')

time = dataset.iloc[:,0]
absolute_accel = dataset.iloc[:,4]

fig, ax = plt.subplots(figsize=(10, 5))

ax.scatter(time, absolute_accel, c='red', marker=".", s=2)

# x and y labels
ax.set_xlabel('Time (seconds)', fontsize = 8)
ax.set_ylabel('Absolute acceleration ($m/s^2$)', fontsize=8)
ax.set_title('Absolute acceleration vs time', fontsize=8)

plt.grid()
plt.show()
