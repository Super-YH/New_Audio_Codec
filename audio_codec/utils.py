import librosa
import numpy as np
import resampy

def point_listenable_mid(mid, side):
    mid_db = librosa.amplitude_to_db(mid)
    side_db = librosa.amplitude_to_db(side)
    mid_point = np.zeros(len(mid), dtype=np.float64)
    side_point = np.zeros(len(side), dtype=np.float64)
    for i in range(len(mid)-1):
        if mid_db[i] < -80:
            mid_point[i] = 0.2
        elif mid_db[i] < -60:
            mid_point[i] = 0.3
        elif mid_db[i] < -40:
            mid_point[i] = 0.5
        elif mid_db[i] < -20:
            mid_point[i] = 0.8
        elif mid_db[i] < -12:
            mid_point[i] = 0.87
        elif mid_db[i] < -6:
            mid_point[i] = 0.99
        if mid_db[i] < side_db[i]:
            mid_db *= (side_db[i] / mid_db[i])
    return mid_point

def point_listenable_side(mid, side):
    mid_db = librosa.amplitude_to_db(mid)
    side_db = librosa.amplitude_to_db(side)
    mid_point = np.zeros(len(mid), dtype=np.float64)
    side_point = np.zeros(len(side), dtype=np.float64)
    for i in range(len(side)):
        if side_db[i] < -80:
            side_point[i] = 0.2
        elif side_db[i] < -60:
            side_point[i] = 0.3
        elif side_db[i] < -40:
            side_point[i] = 0.5
        elif side_db[i] < -20:
            side_point[i] = 0.8
        elif side_db[i] < -12:
            side_point[i] = 0.87
        elif side_db[i] < -6:
            side_point[i] = 0.99
        if side_db[i] < mid_db[i]:
            side_point[i] *= (mid_db[i] / side_db[i])
    return side_point

def band_split(dat, band_num):
    ffted = librosa.stft(dat)
    div = int((ffted.shape[0]-1)/band_num)
    arr = np.zeros([div, ffted.shape[1]], dtype=np.complex128)
    band = []
    for i in range(0, band_num):
        for j in range(ffted.shape[1]):
            sample = ffted[:,j]
            arr[:,j] = sample[i*div:(i+1)*div]
        band.append(librosa.istft(arr, n_fft=div))
    return band

def select_zero_cross(db, zero_cross):
    ##print(zero_cross)
    zero_cross = np.array(zero_cross)[0]
    ##print(zero_cross.shape)
    i = 0
    while True:
        try:
            #print(zero_cross[i+1] - zero_cross[i])
            if (zero_cross[i+1] - zero_cross[i]) < 2048:
                zero_cross = np.delete(zero_cross, i+1)
                continue
            else:
                i += 1
        except:
            break
    return zero_cross

def search_nearest_array(arr_list, arr):
    point = []
    for i in range(len(arr_list)):
        arr_list[i] = np.abs(resampy.resample(np.array(arr_list[i]), len(arr_list[i]), len(arr))).tolist()
    for i in range(len(arr_list)):
        point_s = 0
        for j in range(len(arr)):
            #print(arr[j])
            #print(arr_list[i][j])
            point_s += (arr[j] - arr_list[i][j])
        point.append(point_s)
    #print(np.where(point == np.min(point))[0])
    return np.where(point == np.min(point))[0][0]
