import matplotlib.pyplot as plt
import numpy as np

#ed = np.loadtxt('emotion_regr.csv', delimiter=',', skiprows=1)
#ed = np.loadtxt('emotion_regr_sigmoid.csv', delimiter=',', skiprows=1)
ed = np.loadtxt('emotion_regr_sigmoid_sem.csv', delimiter=',', skiprows=1)

plt.errorbar(ed[:,0], ed[:,1], ed[:,2], linestyle='None', marker='^')
plt.hlines(0.758441432792046, ed[0,0], ed[-1,0], linestyles='dashed', colors='red')
plt.ylim([0,1])
plt.xlabel('Classification threshold')
plt.ylabel('AUC')
plt.title('Regression of subjective emotion response')
plt.show()
