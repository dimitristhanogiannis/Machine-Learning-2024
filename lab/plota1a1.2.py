import numpy as np
import matplotlib.pyplot as plt

mx, my = 5, 10
sx, sy = 3, 4
p = 0.6
c = 0.161

x_values = np.linspace(-5, 15, 400)
y_values = np.linspace(0, 20, 400)
X, Y = np.meshgrid(x_values, y_values)

Z = ((X - mx)**2 / sx**2 +
     (Y - my)**2 / sy**2 -
     2 * p * (X - mx) * (Y - my) / (sx * sy))

plt.contour(X, Y, Z, levels=[c], colors = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('ισοσταθμική καμπύλη άσκησης 1.2')
plt.grid(True)
plt.show()
