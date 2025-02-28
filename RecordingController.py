import requests
import serial
import asyncio
import os
from dotenv import load_dotenv, dotenv_values

# Async API Call to ITAB gate
async def gate_sequence():
    url = "http://192.168.30.139:5000/open_count"

    headers = {
        "Content-Type": "application/json"
    }
    id = 122
    open_count = 1
    data = {
        "id": id,
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

# serial communication with mcu to start recording
async def mcu_recording(gate_type, seq_num):
    port = os.getenv("UART_PORT")
    baudrate = 115200
    timeout = 1
    ser = serial.Serial(port, baudrate, timeout=timeout)
    byte_value = (gate_type << 7) | (seq_num & 0b01111111)
    byte_to_send = bytes([byte_value])
    ser.write(byte_to_send)
    ser.close()
    print(f"Byte in binary: {bin(byte_value)[2:].zfill(8)}")
    print(f"Byte to send: {byte_to_send}")
    mcu_response = await ser.read(size=1)
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
    print("Sequence numbers: 0-127")    
    seq_num = int(input("Enter total sequences to record: "))
    print(os.getenv("UART_PORT"))

    for i in range(1, seq_num+1):
        print(f"Recording sequence {i}")
        # mcu_response = asyncio.create_task(mcu_recording(gate_type, seq_num))
        # gate_response = asyncio.create_task(gate_sequence())
        gate_response = asyncio.create_task(simulate_gate())
        mcu_response = asyncio.create_task(mcu_recording(gate_type, seq_num))
        print({bin(await mcu_response)[2:].zfill(8)})
        print(await gate_response)

    print("Recording completed")

load_dotenv(override=True)
asyncio.run(main())