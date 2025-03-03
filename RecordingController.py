import requests
import serial
import asyncio
import os
from dotenv import load_dotenv, dotenv_values

# Async API Call to ITAB gate
async def gate_sequence():
    url = os.getenv("GATE_URL")

    headers = {
        "Content-Type": "application/json"
    }
    open_count = 1
    data = {
        "id": os.getenv("GATE_ID"),
        "open_count": open_count
    }

    response = await requests.post(
        url,
        headers=headers,
        json = data)

    print("Status Code", response.status_code)
    print("Response", response.text)
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
        raise Exception("No byte received from mcu during demount.")
    
    return mcu_response

def mcu_mount(ser: serial.Serial):
    byte_to_send = bytes([0xFF])

    ser.write(byte_to_send)

    mcu_response = ser.read(size=1)

    # If a timeout is set and no response is received.
    if len(mcu_response) == 0:
        raise Exception("No byte received from mcu during mount.")
    
    return mcu_response


# serial communication with mcu to start recording
async def mcu_recording(ser: serial.Serial, gate_type, seq_num):

    byte_value = (gate_type << 7) | (seq_num & 0b01111111)
    byte_to_send = bytes([byte_value]) 

    print(f"Byte in binary: {bin(byte_value)}")
    print(f"Byte to send: {byte_to_send}")

    ser.write(byte_to_send)

    mcu_response = ser.read(size=1)

    # If a timeout is set and no response is received.
    if len(mcu_response) == 0:
        raise Exception("No byte received from mcu.")
    
    ser.close()

    return mcu_response

# simulates the mcu recording
async def simulate_mcu(gate_type, seq_num):
    print("MCU recording started")
    print(f"Gate Type: {gate_type}, Sequence Number: {seq_num}")
    await asyncio.sleep(6)
    print("MCU recording stopped")
    return 0b10000000

async def main():
    print("Gate types: Good-Gate = 1, Faulty-Gate = 0")
    gate_type = int(input("Enter gate type: "))
    print("Sequence numbers: 0-126")    
    seq_num = int(input("Enter total sequences to record: "))

    # Open serial communication to mcu
    port = os.getenv("UART_PORT")
    baudrate = 115200
    timeout = 15
    ser = serial.Serial(port, baudrate, timeout=timeout)

    mcu_response = mcu_mount(ser)

    if mcu_response == b"\x00":
        raise Exception("MCU Error: Mount error")
    elif mcu_response == b"\xF1":
        print("MCU Success: Mount success")
    else:
        print(f"Unknown response from mcu: {mcu_response}")


    for i in range(seq_num):
        if(seq_num > 126):
            seq_num = 126

        print(f"Recording sequence {i}")
        # mcu_response = asyncio.create_task(mcu_recording(gate_type, seq_num))
        # gate_response = asyncio.create_task(gate_sequence())
        gate_response = asyncio.create_task(simulate_gate())
        mcu_response = asyncio.create_task(mcu_recording(ser, gate_type, i))
        
        print("gate_response", await gate_response)
        await mcu_response

        if mcu_response == b"\xF0":
            print("MCU Success: Recording success")
        elif mcu_response == b"\x01":
            raise Exception("MCU Error: Cannot create wavefile")
        elif mcu_response == b"\x02":
            raise Exception("MCU Error: Cannot close wavefile")
        elif mcu_response == b"\x03":
            raise Exception("MCU Error: Card not mounted")


    mcu_response = mcu_demount(ser)

    if mcu_response == b"\xF2":
        print("MCU Response: Demount success")
    else:
        print(f"Unknown response from mcu: {mcu_response}")

    print("Recording completed")




load_dotenv(override=True)
asyncio.run(main())