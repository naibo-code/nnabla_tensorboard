# From https://github.com/lanpa/tensorboardX/blob/master/tests/test_figure.py

import matplotlib.pyplot as plt
from nnabla_tensorboard import SummaryWriter

plt.switch_backend('agg')

fig = plt.figure()

c1 = plt.Circle((0.2, 0.5), 0.2, color='r')
c2 = plt.Circle((0.8, 0.5), 0.2, color='r')

ax = plt.gca()
ax.add_patch(c1)
ax.add_patch(c2)
plt.axis('scaled')

writer = SummaryWriter()
writer.add_figure('matplotlib', fig)
writer.close()
