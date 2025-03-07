import requests
import serial
import asyncio
import os
import json
from dotenv import load_dotenv, dotenv_values

# Async API Call to ITAB gate
async def gate_sequence():
    url = os.getenv("GATE_URL")

    payload = json.dumps({
    "id": int(os.getenv("GATE_ID")),
    "open_count": 1
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text

# simulates the gate sequence
async def simulate_gate():
    await asyncio.sleep(3)
    print ("Gate opened")
    await asyncio.sleep(3)
    print ("Gate closed")
    return "Done"

def mcu_demount(ser: serial.Serial):
    byte_to_send = bytes([0x7F])

    ser.write(byte_to_send)

    mcu_response = ser.read(size=1)

    # If a timeout is set and no response is received.
    if len(mcu_response) == 0:
        raise Exception("MCU Error: No answer received after demount")
    
    return mcu_response

def mcu_mount(ser: serial.Serial):
    byte_to_send = bytes([0xFF])

    ser.write(byte_to_send)

    mcu_response = ser.read(size=1)

    # If a timeout is set and no response is received.
    if len(mcu_response) == 0:
        raise Exception("MCU Error: No answer received after mount")
    
    return mcu_response


# serial communication with mcu to start recording
async def mcu_recording(ser: serial.Serial, gate_type):

    byte_value = (gate_type << 7)
    byte_to_send = bytes([byte_value]) 

    ser.write(byte_to_send)
    await asyncio.sleep(1)
    mcu_response = ser.read(size=1)

    # If a timeout is set and no response is received.
    if len(mcu_response) == 0:
        raise Exception("MCU Error: No answer received after recording")
    
    return mcu_response

# simulates the mcu recording
async def simulate_mcu(gate_type, seq_num):
    print("MCU recording started")
    print(f"Gate Type: {gate_type}, Sequence Number: {seq_num}")
    await asyncio.sleep(6)
    print("MCU recording stopped")
    return 0b10000000

def open_serial():
 # Open serial communication to mcu
    port = os.getenv("UART_PORT")
    baudrate = 115200
    timeout = 15
    ser = serial.Serial(port, baudrate, timeout=timeout)
    return ser

def close_serial(ser: serial.Serial):
    ser.close()

def get_input_gate_type():
    print("Gate types: Good-Gate = 1, Faulty-Gate = 0")
    gate_type = int(input("Enter gate type: "))
    while gate_type < 0 or gate_type > 1:
        print("Invalid gate_type...")
        gate_type = int(input("Enter gate type: "))
    return gate_type

def get_input_seq_num():
    print("Sequence numbers: 1-1000")    
    seq_num = int(input("Enter total sequences to record: "))
    while seq_num < 1 or seq_num > 1000:
        print("Invalid seq_num...")
        seq_num = int(input("Enter total sequences to record: "))
    return seq_num

def ctl_mount_response(mcu_response):
    if mcu_response == b"\xF1":
        print("MCU Success: Mount success")
        print("------------------------------")
    elif mcu_response == b"\x00":
        raise Exception("MCU Error: Mount error")
    else:
        raise Exception(f"MCU Unknown Response: Mount response {mcu_response}")

def ctl_demount_response(mcu_response):
    if mcu_response == b"\xF2":
        print("MCU Response: Demount success")
        print("------------------------------")
    else:
        raise Exception(f"MCU Unknown Response: Demount response {mcu_response}")

def ctl_recording_response(mcu_response):
        if mcu_response == b"\xF0":
            print("MCU Success: Recording success")
        elif mcu_response == b"\x01":
            raise Exception("MCU Error: Cannot create wavefile")
        elif mcu_response == b"\x02":
            raise Exception("MCU Error: Cannot close wavefile")
        elif mcu_response == b"\x03":
            raise Exception("MCU Error: Card not mounted")
        else:
            raise Exception(f"MCU Unknown Response: Recording response {mcu_response}")

def ctl_gate_response(gate_response):
    if gate_response == "Done":
        print("Gate Success: Gate sequence completed")
    else:
        raise Exception(f"Gate Error: Gate sequence error {gate_response}")

async def main():
    gate_type = get_input_gate_type()
    seq_num = get_input_seq_num()

    print("------------------------------")

    ser = open_serial()

    ctl_mount_response(mcu_mount(ser))

    for i in range(seq_num):

        print(f"Recording sequence {i+1}")
        mcu_response = asyncio.create_task(mcu_recording(ser, gate_type))
        gate_response = asyncio.create_task(gate_sequence())
        # gate_response = asyncio.create_task(simulate_gate())
        # mcu_response = asyncio.create_task(sim(gate_type, seq_num))
        ctl_gate_response(await gate_response)
        ctl_recording_response(await mcu_response)
        print("------------------------------")

    print("Recording completed")

    ctl_demount_response(mcu_demount(ser))

    close_serial(ser)

load_dotenv(override=True)
asyncio.run(main())