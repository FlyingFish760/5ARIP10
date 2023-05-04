import sounddevice as sd
from scipy.io.wavfile import write
import os.path

fs = 44100  # Sample rate
seconds = 10  # Duration of recording
test_day = 'Microphone_Omron_'
file_type = '.wav'

test_type = '17' #Name of test

test_name = test_day + 'test_' + str(test_type) + '_fs' + str(fs) + '_secs' + str(seconds) + file_type

print(test_name)


print("Begin recording")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
print(len(myrecording))
write(test_name, fs, myrecording)  # Save as WAV file

