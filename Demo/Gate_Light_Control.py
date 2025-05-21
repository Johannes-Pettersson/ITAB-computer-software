import requests
import time
import os
import json
from dotenv import load_dotenv


def set_led_color(payload):
    try:
        load_dotenv(override=True)
        url = os.getenv("GATE_URL") + "/set_led_color"
        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=payload, headers=headers, timeout=0.5)
        response.raise_for_status()
    except Exception as e:
        raise e

def gate_blink_sequence(color):
    """
    Sends commands to the gate to turn the light on and off in a sequence.
    :param color: The color to blink (red or green).
    """

    red = 255 if color == "red" else 0
    green = 255 if color == "green" else 0

    error_count = 0
    max_errors = 3

    for i in range(10):
        try:
            payload = {
                "red": int(red) if i % 2 == 0 else int(0),
                "green": int(green) if i % 2 == 0 else int(0),
                "blue": int(0),
                "state": "exit_closed_arm",
            }
            set_led_color(payload)
        except Exception as e:
             error_count += 1
             if error_count >= max_errors:
                print("Max errors reached, stopping the sequence.")
                return
        time.sleep(0.5)

    # Ensure the light is turned off after the sequence
    payload = {
        "red": int(0),
        "green": int(0),
        "blue": int(255),
        "state": "exit_closed_arm",
    }
    set_led_color(payload)

def gate_sequence():
    load_dotenv(override=True)
    url = os.getenv("GATE_URL") + "/open_count"

    payload = json.dumps({
    "id": int(os.getenv("GATE_ID")),
    "open_count": 1
    })
    headers = {
    'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text


if __name__ == "__main__":
     gate_blink_sequence("red")
