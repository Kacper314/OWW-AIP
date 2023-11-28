import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time
import fastdtw



czestotliwosc1, dzwiek1 = wavfile.read("C:/Users/Kacper/Downloads/doors-and-corners-kid_thats-where-they-get-you (1).wav")
czestotliwosc2, dzwiek2 = wavfile.read("C:/Users/Kacper/Downloads/you-walk-into-a-room-too-fast_the-room-eats-you (1).wav")
arr1 = np.array(dzwiek1)
data1 = arr1.flatten()
arr2 = np.array(dzwiek2)
data2 = arr2.flatten()
data11=data1[1::300]
data11 = data11 / (2.**15)
data22=data2[1::300]
data22 = data22 / (2.**15)


def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw = np.zeros((n+1, m+1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw[i, j] = np.inf
    dtw[0][0] = 0

    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(s1[i-1] - s2[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

    return dtw[n][m]


def dtw_window(s1, s2, window):
    n, m = len(s1), len(s2)
    w = np.max([window, abs(n - m)])
    dtw_matrix = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0


    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            dtw_matrix[i, j] = 0

    for i in range(1, n + 1):
        for j in range(np.max([1, i - w]), np.min([m, i + w]) + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            # take last min from a square box
            last_min = np.min([dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]])
            dtw_matrix[i, j] = cost + last_min
    return dtw_matrix, dtw_matrix[n][m]

distance1 = dtw_distance(data11, data22)
dtw, dist = dtw_window(data11, data22, 350)
#distt = dtw1(data11,data22)
print(distance1)
print(dist)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10,10))

ax1.plot(data11+1, label='s1')
ax1.plot(data22, label='s2')
ax1.set(title='Sequences', xlabel='Time', ylabel='Value')
ax1.legend()

im = ax2.imshow(dtw, cmap=plt.cm.binary, interpolation="nearest", origin="lower")
ax2.set(title='DTW Matrix', xlabel='s1', ylabel='s2')

fig.colorbar(im, ax=ax2)


i, j = len(data11)-1, len(data22)-1
path = [(i, j)]
while i > 0 or j > 0:
    prev = min(dtw[i-1][j-1], dtw[i][j-1], dtw[i-1][j])
    if prev == dtw[i-1][j-1]: #match
        i, j = i-1, j-1
    elif prev == dtw[i][j-1]: #deletion
        j -= 1
    elif prev == dtw[i-1][j]:
        i -= 1
    path.append((i, j))

for x_i, y_j in path[1::10]:
    plt.plot([x_i, y_j], [data11[x_i] + 1.5, data22[y_j] - 1.5], c="C7")
plt.plot(np.arange(data22.shape[0]), data22 - 1.5)
plt.plot(np.arange(data11.shape[0]), data11 + 1.5)
print(path)

ax3.set(title='DTW Path', xlabel='Time', ylabel='Value')
ax3.legend()

plt.tight_layout()
plt.show()





