import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time
import multiprocessing
import fastdtw

def dtw_distance(s1, s2):
    n, m = len(s1), len(s2)
    dtw = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        dtw[i][0] = float('inf')
    for j in range(1, m+1):
        dtw[0][j] = float('inf')
    #dtw[0][0] = 0

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

def main():
    czestotliwosc1, dzwiek1 = wavfile.read(
        "C:/Users/Kacper/Downloads/doors-and-corners-kid_thats-where-they-get-you (1).wav")
    czestotliwosc2, dzwiek2 = wavfile.read(
        "C:/Users/Kacper/Downloads/you-walk-into-a-room-too-fast_the-room-eats-you (1).wav")
    #czestotliwosc2, dzwiek2 = czestotliwosc1, dzwiek1
    arr1 = np.array(dzwiek1)
    data1 = arr1.flatten()
    arr2 = np.array(dzwiek2)
    data2 = arr2.flatten()
    data11=data1[1::300]
    data11 = data11 / (2.**15)
    data22=data2[1::300]
    data22 = data22 / (2.**15)

    #Multiprocessing
    num_processes = multiprocessing.cpu_count()
    #print(num_processes)

    
    chunk_size1 = int(len(data11)/num_processes)
    chunk_size2 = int(len(data22)/num_processes)

    # Analiza przedziałów ?????????

    pool = multiprocessing.Pool(processes=num_processes)

    ranges1 = [(i * chunk_size1, (i + 1) * chunk_size1) for i in range(num_processes)]
    ranges2 = [(i * chunk_size2, (i + 1) * chunk_size2) for i in range(num_processes)]
    print(ranges1)
    print(ranges2)
    ranges_test = [(data11[ranges1[i][0]:ranges1[i][1]], data22[ranges2[i][0]:ranges2[i][1]]) for i in range(num_processes)]


    start_time = time.time()
    print(len(data22))
    distance1 = dtw_distance(data11, data22)
    time1 = time.time() - start_time
    print("Bez wątków: ", time1)
    print(distance1)

    start_time1 = time.time()
    dist = pool.starmap(dtw_distance, ranges_test) #lista
    #distance1 = dtw_distance(data11, data22)
    #distance1 = fastdtw.fastdtw(data11, data22)[0]
    time2 = time.time() - start_time1
    print("Z wątkami: ", time2)
    print(sum(dist))



if __name__ == "__main__":
    main()