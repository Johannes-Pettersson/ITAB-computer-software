import re
import ast

INPUT_FILE_NAME = "test.txt"
OUTPUT_FILE_NAME = "recording.wav"

with open(INPUT_FILE_NAME, "r") as f:
    input_data = f.read()


pattern = r"'0xfa', '0x5', '0xfa', '0x5'(.*?)'0xfb', '0x6', '0xfb', '0x6'"

matches = re.findall(pattern, input_data, re.DOTALL)

messages = []

for match in matches:
    try:
        packet = ast.literal_eval("[" + match.strip(", ") + "]")
        messages.append(packet)
    except Exception as e:
        print("Error when evaluating packet:", e)


output_file_data = bytearray()

for i, msg in enumerate(messages):
    if msg[:4] == ['0xfc', '0x8', '0xfc', '0x8']:
        # file length message

        file_len_str = msg[4:]

        file_len_raw_bytes = bytes(int(b, 16) for b in file_len_str)
        
        for i in range(len(file_len_raw_bytes)):
            output_file_data[40+i] = file_len_raw_bytes[i]


        file_len_int = int.from_bytes(file_len_raw_bytes, byteorder='little', signed=False)

        file_with_header_len_int = file_len_int + 44

        file_with_header_len_bytes = file_with_header_len_int.to_bytes(4, byteorder='little')

        for i in range(len(file_with_header_len_bytes)):
            output_file_data[4+i] = file_with_header_len_bytes[i]


        # File should be ended, therefore break:
        break
        
    else:
        # ordinary sound data message
        for byte_string in msg:
            output_file_data.extend(bytearray([int(byte_string, 16)]))
        

with open(OUTPUT_FILE_NAME, "wb") as f:
    f.write(output_file_data)