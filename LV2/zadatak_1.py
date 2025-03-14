import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 3, 3, 2,1])
y = np.array([1,1,2,2,1])
plt.figure()
plt.plot(x,y,'r', linewidth=5, marker="*", markersize=20)
plt.axis([0,4,0,4])
plt.xlabel('x os')
plt.ylabel('y os')
plt.title('Primjer')
plt.show()