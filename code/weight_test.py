import numpy as np

x = np.array([75, 138, 679])
y = np.array([3.45, 2.7, 0.3])
a = np.polyfit(x, np.log(y),1)
demand_erraticity = 140
print(a)
print(round(np.exp(a[0] *demand_erraticity + a[1])))

print(isinstance(int(np.round(2.99)), int))