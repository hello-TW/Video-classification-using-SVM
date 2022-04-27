from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

#nature = np.array([[36.36363636,18.14], [27.27272727,8.5], [12.5,18.4], [83.33333333, 0.5], [66.66666667,5.3], [20, 6.7], [42.85714286,6.4], [71.42857143,6.6], [33.33333333,18], [100,5.8],
#                    [100,1.59], [100,5.6], [0,0.7], [55.55555556,1.2], [83.33333333,1.6], [100,1.68], [100,1.39], [100,1.55], [75,29.7] ,[100, 1.55]])
# ainmal = np.array([[100,1.59], [100,5.6], [0,0.7], [55.55555556,1.2], [83.33333333,1.6], [100,1.68], [100,1.39], [100,1.55], [75,29.7] ,[100, 1.55]])
data = np.array([[36.36363636,18.14], [27.27272727,8.5], [12.5,18.4], [83.33333333, 0.5], [66.66666667,5.3], [20, 6.7], [42.85714286,6.4], [71.42857143,6.6], [33.33333333,18], [100,5.8],
                    [100,1.59], [100,5.6], [0,0.7], [55.55555556,1.2], [83.33333333,1.6], [100,1.68], [100,1.39], [100,1.55], [75,29.7] ,[100, 1.55]])

y1 = np.ones(shape=(10,), dtype=np.int8) * (-1)
y2 = np.ones(shape=(10,), dtype=np.int8)
y = np.concatenate((y1,y2), axis=0)

cl = svm.SVC(kernel='linear', C=10000)
cl.fit(data, y)

print(cl.support_vectors_)
plt.scatter(data[0:9,0], data[0:9,1],label='Nature')
plt.scatter(data[10:19,0], data[10:19,1],label='Animal')

x = np.linspace(0,100,100)
y = np.linspace(0,30,100)
X,Y = np.meshgrid(x,y)
xy = np.vstack([X.ravel(), Y.ravel()]).T
Z = cl.decision_function(xy).reshape(X.shape)
plt.contour(X,Y,Z, colors='k', levels=[-1,0,1],alpha=0.5, linestypes=['--','-','---'])
plt.xlabel('Emotion')
plt.ylabel('Time(10 minute)')
plt.legend()
plt.show()