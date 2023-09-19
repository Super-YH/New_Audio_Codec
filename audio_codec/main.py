import sys
import time
import os
import soundfile as sf
import core

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

def main():
    if len(sys.argv) == 1:
        print(sys.argv[0] + " mode input output (quality)")
        return 0
    if sys.argv[1] == "enc":
        dat, fs = sf.read(sys.argv[2])
        try:
            quality = int(sys.argv[4])
        except:
            quality = 4
        print("Processing...")
        playback_time = dat.shape[0] / fs
        now_time = time.time()
        dat = core.encode(dat, fs, quality)
        f = open(sys.argv[3], "wb")
        f.write(dat)
        f.close()
        proc_time = time.time() - now_time
        print("Processing Time: " + str(int(proc_time)) + "sec")
        print("Bitrate: " + str(int(len(dat)*8/playback_time/1000*100)/100) + "kbps")
        print("Finish!")
    elif sys.argv[1] == "dec":
        f = open(sys.argv[2], "rb")
        dat = f.read()
        f.close()
        print("Processing...")
        now_time = time.time()
        dat, fs = core.decode(dat)
        sf.write(sys.argv[3], dat, fs, format="WAV")
        proc_time = time.time() - now_time
        print("Processing Time: " + str(int(proc_time)) + "sec")
        print("Finish!")
    else:
        print(sys.argv[0] + " mode input output (quality)")
    return 0

if __name__ == '__main__':
    main()
