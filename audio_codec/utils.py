import librosa
import numpy as np
import resampy
import struct

def func_1(arr):
    ###print(arr)
    new_arr = []
    b = []
    a = 0
    for i in arr:
        if np.sign(i) == 1:
            new_arr.append(np.abs(i)*2)
        elif np.sign(i) == -1:
            new_arr.append(np.abs(i)*2+1)
        else:
            pass
    ##print(new_arr)
    new_arr = int_to_bool(new_arr)
    ##print(len(new_arr))
    ###print(new_arr)
    return new_arr

def func_2(arr):
    ###print(arr)
    new_arr = []
    a = bool_to_int(arr)
    for i in a:
        f = int(i/2)
        if i%2 == 0:
            new_arr.append(f)
        else:
            new_arr.append(f*-1)
    ###print(new_arr)
    return new_arr

def int_to_bool(int_):
    bytes_arr = b""
    for i in range(len(int_)):
        if int_[i] > 255:
            int_[i] = 255
        int_[i] = struct.pack("@B", int(int_[i]))
    for i in int_:
        bytes_arr += bytes(i)
    #print(len(bytes_arr))
    return bytes_arr

def bool_to_int(bool_):
    arr = []
    for i in bool_:
        arr.append(i)
    ##print(arr)
    return arr

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
    for i in range(band_num):
        for j in range(ffted.shape[1]):
            sample = ffted[:,j]
            arr[:,j] = sample[i*div:(i+1)*div]
        print(librosa.istft(arr))
        band.append(librosa.istft(arr))
        arr = np.zeros([div, ffted.shape[1]], dtype=np.complex128)
    return band

def bands_mix(dat, length):
    band_num = len(dat)
    #print(band_num)
    ffted_a = librosa.stft(np.array(dat[0]), n_fft=int(1024/band_num))
    ffted = np.zeros([1024, int(length/1024*2)], dtype=np.complex128)
    for i in range(band_num):
        ffted_s = librosa.stft(np.array(dat[i]), n_fft=int(1024/band_num*2))
        for j in range(ffted_s.shape[1]):
            ffted[:,j][i*int(1024/band_num):(i+1)*int(1024/band_num)] = ffted_s[:,j][:int(1024/band_num)]
    return librosa.istft(ffted)

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
