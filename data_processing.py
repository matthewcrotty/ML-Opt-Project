import scipy.io.wavfile as wf
import python_speech_features as sf
import pickle 
import os
x0, x1, x2, x3, x4, x5, x6, x7, x8, x9 = [], [], [], [], [], [], [], [], [], []
digits = {'0': x0, '1': x1, '2': x2, '3': x3, '4': x4, '5': x5, '6': x6, '7': x7, '8': x8, '9': x9}
for root, dir, files in os.walk('./data'):
    for file in files:
        if file[3] == '2':
            break
        sample_rate, sigs = wf.read(os.path.join(root, file))
        mfcc = sf.mfcc(sigs, sample_rate, nfft=2048)
        digits[file[0]].append(mfcc)

for d in digits.keys():
    f = open(str(digits[d])+'.pkl', 'wb')
    p = pickle.dumps(digits[d], f, -1)