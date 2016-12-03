import numpy as np
import matplotlib.pyplot as plt

# N = 10
# data = np.random.random((N, 4))
# labels = ['point{0}'.format(i) for i in range(N)]
# plt.subplots_adjust(bottom = 0.1)
# plt.scatter(
#     data[:, 0], data[:, 1], marker = 'o', c = data[:, 2], s = data[:, 3]*1500,
#     cmap = plt.get_cmap('Spectral'))
# for label, x, y in zip(labels, data[:, 0], data[:, 1]):
#     plt.annotate(
#         label,
#         xy = (x, y), xytext = (-20, 20),
#         textcoords = 'offset points', ha = 'right', va = 'bottom',
#         bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#         arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

# plt.show()

# plt.plot([0.3, -0.6, 0.2, 0.67], [1, 1, 1, 1], 'go',
#          [0.36, -0.2, -0.27], [2, 2, 2], 'yo')
# plt.axis([-1, 1, 0, 4])
# plt.xlabel('Tendency Score')
# plt.show()

colors = ['ro', 'bo', 'go', 'yo']

data_x = [0.3, -0.6, 0.2, 0.67, 0.36, -0.2, -0.27]
data_y = [1, 1, 1, 1, 2, 2, 2]
labals = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
y_value = 1

plt.scatter(data_x, data_y, marker='o', c='lightgrey', s=200)
# plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_ticks( [ 1, 2, 3 ] )
plt.gca().axes.get_yaxis().set_ticklabels( ['tom','dick','harry'])
plt.show()
