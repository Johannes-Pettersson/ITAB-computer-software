
import RPi.GPIO as GPIO
from time import sleep
import spidev


spi = spidev.SpiDev()

spi.open(0,0)
spi.max_speed_hz = 1000000
spi.mode = 0

FILE_PIN = 17
WRITE_PIN = 27

file_started = False
writing_to_file = False

def file_pin_callback(channel):
    global file_started
    if GPIO.input(FILE_PIN) == GPIO.HIGH:
        file_started = True
    else:
        file_started = False

def write_pin_callback(channel):
    global writing_to_file
    if GPIO.input(WRITE_PIN) == GPIO.HIGH:
        writing_to_file = True
    else:
        writing_to_file = False

GPIO.setmode(GPIO.BCM)

GPIO.setup(FILE_PIN, GPIO.IN)
GPIO.setup(WRITE_PIN, GPIO.IN)


GPIO.add_event_detect(FILE_PIN, GPIO.BOTH, callback=file_pin_callback)
GPIO.add_event_detect(WRITE_PIN, GPIO.BOTH, callback=write_pin_callback)

message = []

try: 
    while not file_started:
        sleep(0.01)
    # File started
    print("file started")

    while file_started:
        if writing_to_file:
            response = spi.xfer2([0x00]*4096)
            message.extend(response)
        else:
            sleep(0.01)


finally:
    with open("test.txt", "w") as f:
        for i in range(len(message)):
            message[i] = hex(message[i])

        f.write(str(message))

    spi.close()
    GPIO.cleanup()