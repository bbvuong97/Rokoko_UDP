import socket
import json

# Socket settings
local_ip = "127.0.0.1"
local_port = 14042

# Create a UDP socket to receive Rokoko data
receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
receiver_socket.bind((local_ip, local_port))

print(f"Listening for UDP data from Rokoko Studio on {local_ip}:{local_port}...")

try:
    while True:
        # Receive data from Rokoko Studio
        data, addr = receiver_socket.recvfrom(4096)  # Buffer size
        print(f"Data received from {addr}")

        # Decode the raw JSON data
        raw_json = data.decode('utf-8')
        print("Raw JSON Data:", raw_json)  # Print raw JSON string for debugging

        # Parse JSON data
        try:
            parsed_data = json.loads(raw_json)
            print("Parsed JSON Data:", parsed_data)  # Print the parsed JSON object
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    # Close the socket
    receiver_socket.close()
    print("Socket closed.")
