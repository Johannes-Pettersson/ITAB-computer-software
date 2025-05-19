import requests
import time
import os
from dotenv import load_dotenv


def set_led_color(payload):
    try:
        load_dotenv(override=True)
        url = os.getenv("GATE_URL")
        print("url: ", url)

        headers = {
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=payload, headers=headers, timeout=0.5)
        if response.status_code != 200:
            print(f"Failed to send light command: {response.status_code}")
    except requests.exceptions.RequestException as e:
                print(f"Error sending light command: {e}")

def gate_blink_sequence(color):
    """
    Sends commands to the gate to turn the light on and off in a sequence.
    :param color: The color to blink (red or green).
    """

    red = 255 if color == "red" else 0
    green = 255 if color == "green" else 0

    print(f"Starting blink sequence with color: {color}")


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
             print(f"Error when sending payload: {e}")
        time.sleep(0.5)

    # Ensure the light is turned off after the sequence
    payload = {
        "red": int(0),
        "green": int(0),
        "blue": int(0),
        "state": "exit_closed_arm",
    }
    set_led_color(payload)


if __name__ == "__main__":
     gate_blink_sequence("red")