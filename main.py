import numpy as np
from hmmlearn import hmm
# https://hmmlearn.readthedocs.io/en/latest/tutorial.html#available-models

model = hmm.GaussianHMM(n_components=10, covariance_type="full")

import scipy.io.wavfile as wf
import python_speech_features as sf

import os
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = [], [], [], [], [], [], [], [], [], []
digits = {'0': x0, '1': x1, '2': x2, '3': x3, '4': x4, '5': x5, '6': x6, '7': x7, '8': x8, '9': x9}
for root, dir, files in os.walk('./data'):
    for file in files:
        sample_rate, sigs = wf.read(os.path.join(root, file))
        mfcc = sf.mfcc(sigs, sample_rate, nfft=2048)
        digits[file[0]].append(mfcc)

print(x0)
# sample_rate, sigs = wf.read('./data/01/0_01_0.wav')
# mfcc = sf.mfcc(sigs, sample_rate)
# print(mfcc.shape)

# 10 Hmm models for each class
# run data point through all models to find best fit