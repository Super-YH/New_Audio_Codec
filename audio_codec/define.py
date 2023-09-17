class AUDIO:
    dat = []
    fs = 0
    mid_start = 0
    mid_end = 0
    side_start = 0
    side_end = 0
    curve = [[1,0.1], [1,0.5,0.1], [1, 0.5], [0.7, 1], [0.1, 1], [0.1, 0.5, 1], [0.5, 1], [0.7, 1]]
    band_num = 64
    version = 0.01
