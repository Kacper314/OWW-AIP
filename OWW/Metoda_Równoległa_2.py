import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time
import multiprocessing
import fastdtw


def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        dtw[i][0] = float('inf')
    for j in range(1, m + 1):
        dtw[0][j] = float('inf')
    # dtw[0][0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw[i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])

    return dtw[n][m]


def main():
    czestotliwosc1, dzwiek1 = wavfile.read(
        "doors-and-corners-kid_thats-where-they-get-you.wav")
    czestotliwosc2, dzwiek2 = wavfile.read(
        "you-walk-into-a-room-too-fast_the-room-eats-you.wav")
    # czestotliwosc2, dzwiek2 = czestotliwosc1, dzwiek1
    arr1 = np.array(dzwiek1)
    data1 = arr1.flatten()
    arr2 = np.array(dzwiek2)
    data2 = arr2.flatten()
    data11 = data1[1::300]
    data11 = data11 / (2. ** 15)
    data22 = data2[1::300]
    data22 = data22 / (2. ** 15)

    num_processes = 7
    print(multiprocessing.cpu_count())
    chunk_size1 = int(len(data11) / num_processes)
    chunk_size2 = int(len(data22) / num_processes)

    pool = multiprocessing.Pool(processes=num_processes)

    ranges1 = [(i * chunk_size1, (i + 1) * chunk_size1) for i in range(num_processes)]
    ranges2 = [(i * chunk_size2, (i + 1) * chunk_size2) for i in range(num_processes)]
    print(ranges1)
    print(ranges2)
    start_time = time.time()

    mins=0
    for j in range(num_processes):
        ranges_test = [(data11[ranges1[j][0]:ranges1[j][1]], data22[ranges2[i][0]:ranges2[i][1]]) for i in range(num_processes)]
        dist = pool.starmap(dtw_distance, ranges_test)  # lista
        mins += min(dist)
        print(mins)

    time2 = time.time() - start_time
    print("Z wątkami: ", time2)
    print(mins)

if __name__ == "__main__":
    main()