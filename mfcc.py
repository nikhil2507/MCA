# https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc
# http://kom.aau.dk/group/04gr742/pdf/MFCC_worksheet.pdf
# https://ijirae.com/volumes/vol1/issue10/27.NVEC10086.pdf
# https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html


import numpy as np
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt

sr, signal = scipy.io.wavfile.read('test.wav') 

def Framing(size=0.025, step = 0.01, p = 0.97):
    y = np.append(signal[0], signal[1:] - p * signal[:-1])
    f_length = int(round(size * sr))
    f_step = int(round(step * sr))

    return (f_length, f_step, y)

f_length, f_step, y = Framing()
signal_length = len(y)
# print(f_length, f_step)

def Padding(signal_length, f_length, f_step):
    x = np.abs(signal_length - f_length) / f_step
    X = float(x)
    num = int(np.ceil(X))

    # Padding
    padLength = num * f_step + f_length
    zeros = np.zeros((padLength - signal_length))
    pad = np.append(y, zeros)

    tile1 = np.tile(np.arange(0, num * f_step, f_step), (f_length, 1)).T
    tile2 = np.tile(np.arange(0, f_length), (num, 1))

    ind = tile1 + tile2

    padFrames = pad[ind.astype(np.int32, copy = False)]

    return padFrames

padFrames = Padding(signal_length, f_length, f_step)
# print(padFrames)

def Window(win, f_length):
    n = np.arange(0, f_length)
    win *= 0.54 - 0.46 * np.cos((2 * np.pi * n) / (f_length - 1)) 
    
    return win

HamWindow = Window(padFrames, f_length)
# plt.imshow(HamWindow)
# plt.savefig('Ham.png')

# NFFT = 512
def Power(HamWindow, NFFT = 512):
    mag = np.absolute(np.fft.rfft(HamWindow, NFFT))
    power = ((1.0 / NFFT) * ((mag) ** 2))

    return mag, power

mag, power = Power(HamWindow)
# print(mag, power)

def Mel(sr, nfilt=40):
    low = 0
    high = (2595 * np.log10(1 + (sr / 2) / 700))
    mel = np.linspace(low, high, nfilt + 2)
    
    return mel

mel = Mel(sr)
# print(mel)

def Hertz(mel, sr, NFFT = 512):
    hz = (700 * (10**(mel / 2595) - 1))
    bin = np.floor((NFFT + 1) * hz / sr)

    return bin

bin = Hertz(mel, sr)
# print(bin)

def FilterBanks(bin, nfilt = 40, NFFT = 512 ):
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        left = int(bin[m - 1])  
        center = int(bin[m])            
        right = int(bin[m + 1])    

        for k in range(center, right):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

        for k in range(left, center):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])

    
    return fbank

fbank = FilterBanks(bin, nfilt = 40, NFFT = 512 )

filter_banks = np.dot(power, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
filter_banks = 20 * np.log10(filter_banks) 

def MFCC(filter_banks,ceps = 12, lifter = 22):

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (ceps + 1)]
    (l, b) = mfcc.shape
    n = np.arange(b)
    lift = 1 + (lifter / 2) * np.sin(np.pi * n / lifter)
    mfcc *= lift

    return mfcc

mfcc = MFCC(filter_banks)

plt.figure()
plt.imshow(mfcc)
plt.savefig('mfcc.png')