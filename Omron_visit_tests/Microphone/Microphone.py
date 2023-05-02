import sounddevice as sd
from scipy.io.wavfile import write
import os.path

save_path = "C:\Users\hugog\OneDrive - TU Eindhoven\TUE\Project\software\5ARIP10\Omron_visit_tests\Test_results\Microphone"
print("test")
fs = 44100  # Sample rate
seconds = 5  # Duration of recording
test_day = 'Microphone_Omron_Test_'
file_type = '.wav'

test_type = '1' #Name of test

test_name = test_day + 'test_' + str(test_type) + '_fs' + str(fs) + '_secs' + str(seconds) + file_type

print(test_name)


print("Begin recording")
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
sd.wait()  # Wait until recording is finished
print(len(myrecording))
write(test_name, fs, myrecording)  # Save as WAV file

