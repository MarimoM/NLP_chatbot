import matplotlib.pyplot as plt

error_threshold = [10, 20, 30, 40, 50, 60, 70, 80]
tikslumas_NB = [90, 90, 90, 92, 92, 84, 56, 35]
tikslumas_KNN = [87, 88, 88, 89, 89, 75, 75, 59]
tikslumas_MLP = [89, 91, 90, 90, 94, 93, 90, 87]
tikslumas_SVM = [90, 90, 92, 92, 90, 86, 84, 78]

fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(2,2)
ax1.plot(error_threshold, tikslumas_NB, marker='o', color='r')
ax2.plot(error_threshold, tikslumas_KNN, marker='o', color='b')
ax3.plot(error_threshold, tikslumas_MLP, marker='o', color='g')
ax4.plot(error_threshold, tikslumas_SVM, marker='o', color='m')

ax1.set_title('Naivusis Bajesas')
ax2.set_title('K arčiausių kaimynų')
ax3.set_title('Daugiasluoksnis perceptronas')
ax4.set_title('Atraminių vektorių mašina')

for ax in fig.get_axes():
    ax.set(xlabel='Riba (%)', ylabel='Tikslumas (%)')
    ax.axis(ymin=19,ymax=101)

ax1.grid()
ax3.grid()
ax2.grid()
plt.grid()

fig.suptitle('Klasifikavimo riba')
plt.show()
