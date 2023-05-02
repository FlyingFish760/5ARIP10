# Testing at Omron

- [Microphone](#Microphone)
- [Vibration](#Vibration)

## Microphone

This explains how to use the program Microphone.py to read the [microphone](https://www.bax-shop.nl/usb-microfoons/devine-m-mic-usb-bk-condensatormicrofoon-zwart?gclsrc=aw.ds#productinformatie) and save the recorded audio as a .wav file. 

Open the program in any IDE and install necassary libraries in the used environment.

```
pip install sounddevice
pip install scipy
```
Adjust the following settings according to the test you want to run.

```
fs, Sample rate in Hz, 1 - 441000
seconds, number of seconds the microphone records
test_day
test_type
```

navigate to the correct directory

```
cd "directory where Microphone.py is stored"
```

Run the program
```
python Microphone.py
```

This saves the recorded file in the current directory as a .wav file.


## Vibration

The programs used and wiring can be found in the following [git](https://github.com/nagimov/adxl345spi). The program used save the file is called Vibration.py. Make a connection with the RaspberryPi following this [guide](https://www.tomshardware.com/reviews/raspberry-pi-headless-setup-how-to,6028.html). When everything is set up you can open Vibration.py in VNC viewer and change the following paramaters.

```
fs, Sample rate in Hz, 1 - 3200
seconds, number of seconds the vibration sensor records
test_day
test_type
```

This saves the output file in the github folder on the raspberry Pi. To get it to your own laptop either use the build in file transfer from [VNC viewer](https://help.realvnc.com/hc/en-us/articles/360002250477-Transferring-Files-Between-Computers-) or push the file using github.
