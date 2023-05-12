# Getting the measurements
First an explanation on how to set up the external sensors is given. Afterwards the handeling of the software of thesetup is discussed. Here it is included how it should be startet up, how measurements can be exported and how changes in the input signal can be applied.

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

The programs used and wiring can be found in the following [git](https://github.com/nagimov/adxl345spi). The address of the RaspberryPi is raspberry or   The program used save the file is called Vibration.py. Make a connection with the RaspberryPi following this [guide](https://www.tomshardware.com/reviews/raspberry-pi-headless-setup-how-to,6028.html). When everything is set up you can open Vibration.py in VNC viewer and change the following paramaters.

```
fs, Sample rate in Hz, 1 - 3200
seconds, number of seconds the vibration sensor records
test_day
test_type
```

To run the program
```
cd 5ARIP10/Omron_visit_tests/Accelerometer
python Vibration.py
```

This saves the output file in the github folder on the raspberry Pi. To get it to your own laptop either use the build in file transfer from [VNC viewer](https://help.realvnc.com/hc/en-us/articles/360002250477-Transferring-Files-Between-Computers-) or push the file using github.



## Using the setup at InnoSpace

Software: dSpace controlDesk

Steps for executing
```
1. Voedingen aan, 
2. Rechter schakelaar uit, 
3. Batterij aangesloten
4. Go to the folder E:/5XWD0_software
5. Run the 'HIL_EM init.m' with (MATLAB)
6. Open the simulink model located in the same folder (MATLAB)
7. Keep Matlab open and go the Desktop screen
8. Open the Dspace software
9. Open the '5XWD0_project' that should also be in recent projects
10. Set online
11. Turn the right switch on, a green light should light up
12. Check if the 'Driver/u_ref' is at 0 otherwise reset LEM, while no currect in running --> IMPORTANT STEP
13. Set all variables to the desired value and press enter. 
14. Start measurement to see the data live
```

Steps for saving the data
```
1. Go to 'Measurement configuration' 
2. Click on the red button that states something like 'start instant recording'
3. Manually end the recording
4. The measurement can be exporded by moving to the 'project' tab, right click on the latest measurement. It should be exported as .csv preferably.
5. The X is the time and the Y is the other outputed data 
```

Steps to change the input signal
```
1. Go offline in the dspace environment (click on 'go offline' or press ctr+F8)
2. Go to the Simulink model
3. Make any desirable changes on how the input 'w_ref' is shaped. Use building blocks from simulink (e.g. block signal, ramp, step etc...)
Initialise any new variables at 0. 
4. Save and build the model by pressing ctrl+s (for saving) and ctrl+b (for building). Be in the correct folder when building
5. Go back to dSpace
6. Go online again in dSpace. 
7. Go for 'Yes' when the popup asks for applying changes on the variables
8.. If any new variables are added in the Simulink Model, add instruments with panel on the right to set the values.
9. Set new values with the instruments and press enter to see the changes in the input signal
```


sample freq:t_step variable    Modeling model settings 


Opening Matlab for the first time, select the following:
1: Kies rti1104
2: Set preferences automaticaly