import os


fs = 3200
seconds = 10
test_day = 'Accelerometer_Omron'
file_type = '.csv'

test_type = '9'

test_name = test_day + '_test' + str(test_type) + '_fs' + str(fs) + '_secs' + str(seconds) + file_type


os.system(f'sudo adxl345spi -t {seconds} -f {fs} -s {test_name}')

