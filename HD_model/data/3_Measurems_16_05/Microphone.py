import sounddevice as sd
from scipy.io.wavfile import write
import os.path

fs = 44100  # Sample rate
seconds = 60  # Duration of recording
test_day = ''
file_type = '.wav'
speed = '60'

test_type = '6' #Name of test

test_name = test_day + 'test_' + str(test_type) + '_speed_' + str(speed) + '_fs' + str(fs) + '_secs' + str(seconds) + file_type

print(test_name)


print("Begin recording")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
print(len(myrecording))
write(test_name, fs, myrecording)  # Save as WAV file

