import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 100)
y = x ** 2

plt.figure(num=3, figsize=(18, 18))
plt.plot(x, y)
plt.show()
