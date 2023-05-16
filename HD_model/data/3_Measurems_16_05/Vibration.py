import os


fs = 3200
seconds = 60
test_day = ''
file_type = '.csv'
speed = '60'
test_type = '6'

test_name = 'test_' + str(test_type) + '_speed_' + str(speed) + '_fs' + str(fs) + '_secs' + str(seconds) + file_type


os.system(f'sudo adxl345spi -t {seconds} -f {fs} -s {test_name}')

