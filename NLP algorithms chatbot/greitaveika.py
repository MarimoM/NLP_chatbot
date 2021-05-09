import numpy as np
import matplotlib.pyplot as plt

NB = [15, 7]
KNN = [1, 11]
MLP = [379, 12]
SVM = [157, 94]

apmokymo_laikas = [15, 1, 279, 157]
testavimo_laikas = [7, 10, 12, 94]

pavadinimai = ['NB', 'KAK', 'DP', 'AVM']

x = np.arange(len(pavadinimai))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, apmokymo_laikas, width, label='Apmokymas')
rects2 = ax.bar(x + width/2, testavimo_laikas, width, label='Testavimas')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Laikas (ms)')
ax.set_title('Modelių greitaveika')
ax.set_xticks(x)
ax.set_xticklabels(pavadinimai)
ax.legend()

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()