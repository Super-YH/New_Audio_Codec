import librosa
import numpy as np
import scipy
import tqdm
import pickle
import pickletools
import pyppmd
import resampy
import utils
import define

def encode(dat, fs, quality):
    audio = define.AUDIO()
    audio.fs = fs
    mid = (dat[:,0] + dat[:,1]) / 2
    side = (dat[:,0] - dat[:,1]) / 2
    quantized = []
    for i in range(len(mid)):
        if np.abs(mid[i]) < 0.0001:
            audio.mid_start += 1
        else:
            break
        if i > 10*fs:
            break
    for i in range(len(mid)):
        if np.abs(mid[i*-1]) < 0.0001:
            audio.mid_end += 1
        else:
            break
        if i > 10*fs:
            break
    for i in range(len(side)):
        if np.abs(side[i]) < 0.0001:
            audio.side_start += 1
        else:
            break
        if i > 10*fs:
            break
    for i in range(len(side)):
        if np.abs(side[i*-1]) < 0.0001:
            audio.side_end += 1
        else:
            break
        if i > 10*fs:
            break
    mid_band = np.array(utils.band_split(mid, audio.band_num))
    side_band = np.array(utils.band_split(side, audio.band_num))
    mid_db = librosa.amplitude_to_db(mid)
    side_db = librosa.amplitude_to_db(side)
    mid_point = np.zeros([audio.band_num, len(mid_band[0])])
    side_point = np.zeros([audio.band_num, len(side_band[0])])
    ###print(audio.band_num)
    for i in range(audio.band_num):
        ###print(i)
        mid_point[i] = utils.point_listenable_mid(mid_band[i], side_band[i])
        side_point[i] = utils.point_listenable_side(mid_band[i], side_band[i])
    zero_cross_mid = []
    zero_cross_side = []
    for i in range(audio.band_num):
        ##print(mid_band[i].tolist())
        ##print(len(mid_band[i]))
        zero_cross_mid.append(np.nonzero(librosa.zero_crossings(mid_band[i], threshold=0.000001)))
        zero_cross_side.append(np.nonzero(librosa.zero_crossings(side_band[i], threshold=0.000001)))
    ###print(zero_cross_mid)
    for i in range(audio.band_num):
        zero_cross_mid[i] = utils.select_zero_cross(mid_db, zero_cross_mid[i])
        zero_cross_side[i] = utils.select_zero_cross(side_db, zero_cross_side[i])
    bands_quan = []
    for i in tqdm.tqdm(range(audio.band_num)):
        chunk = []
        #print((zero_cross_mid[i]))
        for j in tqdm.tqdm(range(len(zero_cross_mid[i])-1)):
            mode_mid = 0
            ##print(zero_cross_mid[i][j+1])
            sample = mid_band[i]
            sample = sample[zero_cross_mid[i][j]:zero_cross_mid[i][j+1]]
            point_mid = mid_point[i]
            point_mid = point_mid[zero_cross_mid[i][j]:zero_cross_mid[i][j+1]]
            ffted_mid = scipy.fft.dst(sample)
            ffted_mid_db = np.abs(librosa.power_to_db(ffted_mid))
            if np.min(np.abs(np.diff(np.abs(ffted_mid_db)))) < 3:
                #mode_mid = 1
                mode_mid = 0
            if np.max(point_mid) < 0.2:
                mode_mid = 2
            if np.max(point_mid) == 0:
                mode_mid = 3
            if mode_mid == 0:
                chunk.append(mode_mid)
                sign_mid = np.sign(ffted_mid)
                ffted_point_mid = np.abs(scipy.fft.dst(point_mid))
                #print(utils.search_nearest_array(audio.curve, (1 / ffted_point_mid)))
                curve_num = int(utils.search_nearest_array(audio.curve, (ffted_point_mid)))
                select_curve_mid = audio.curve[curve_num]
                select_curve_mid = 1 - np.abs(resampy.resample(np.array(select_curve_mid), len(select_curve_mid), len(ffted_mid)))
                #print(select_curve_mid)
                ffted_mid_db *= select_curve_mid
                #print(ffted_mid_db)
                chunk.append(np.rint(ffted_mid_db*sign_mid).tolist())
                chunk.append(curve_num)
            elif mode_mid == 1:
                chunk.append(mode_mid)
                chunk.append(np.hstack([int(ffted_mid_db[0]), np.diff(np.rint(ffted_mid_db))]).tolist())
                chunk.append(curve_num)
            elif mode_mid == 2:
                chunk.append(mode_mid)
            elif mode_mid == 3:
                chunk.append(mode_mid)
        bands_quan.append(chunk)
    quantized.append(bands_quan)
    bands_quan = []
    for i in tqdm.tqdm(range(audio.band_num)):
        chunk = []
        #print((zero_cross_side[i]))
        for j in tqdm.tqdm(range(len(zero_cross_side[i])-1)):
            mode_side = 0
            ##print(zero_cross_side[i][j+1])
            sample = side_band[i]
            sample = sample[zero_cross_side[i][j]:zero_cross_side[i][j+1]]
            point_side = side_point[i]
            point_side = point_side[zero_cross_side[i][j]:zero_cross_side[i][j+1]]
            ffted_side = scipy.fft.dst(sample)
            ffted_side_db = np.abs(librosa.power_to_db(ffted_side))
            if np.min(np.abs(np.diff(ffted_side_db))) < 3:
                mode_side = 0
            if np.max(point_side) < 0.2:
                mode_side = 2
            if np.max(point_side) == 0:
                mode_side = 3
            if mode_side == 0:
                chunk.append(mode_side)
                sign_side = np.sign(ffted_side)
                ffted_point_side = np.abs(scipy.fft.dst(point_side))
                #print(utils.search_nearest_array(audio.curve, (1 / ffted_point_side)))
                select_curve_side = audio.curve[int(utils.search_nearest_array(audio.curve, (ffted_point_side)))]
                select_curve_side = np.abs(resampy.resample(np.array(select_curve_side), len(select_curve_side), len(ffted_side)))
                ffted_side_db *= select_curve_side
                chunk.append(np.rint(ffted_side_db*sign_side).tolist())
            elif mode_side == 1:
                chunk.append(mode_side)
                chunk.append(np.hstack([int(ffted_side_db[0]), np.diff(np.rint(ffted_side_db))]).tolist())
            elif mode_side == 2:
                chunk.append(mode_side)
            elif mode_side == 3:
                chunk.append(mode_side)
        #print(chunk)
        bands_quan.append(chunk)
    quantized.append(bands_quan)
    audio.dat = quantized
    bytes_array = pickle.dumps(audio)
    return pyppmd.compress(pickletools.optimize(bytes_array))

def decode(dat):
    audio = pickle.loads(pyppmd.decompress(dat))
    samples = []
    bands = []
    mid = audio.dat[0]
    side = audio.dat[1]
    for i in range(len(mid)):
        
