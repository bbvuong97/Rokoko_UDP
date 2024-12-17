import socket
import json

# Settings
local_ip = "127.0.0.1"  # IP Rokoko is streaming to (your local machine)
local_port = 14041       # Port Rokoko is streaming to

# Create a TCP socket to receive Rokoko data
receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receiver_socket.bind((local_ip, local_port))
receiver_socket.listen(1)

print(f"Listening for data from Rokoko Studio on {local_ip}:{local_port}...")

try:
    # Accept Rokoko connection
    conn, addr = receiver_socket.accept()
    print(f"Connection established with {addr}")

    while True:
        # Receive data from Rokoko Studio
        data = conn.recv(4096)  # Buffer size
        if not data:
            print("No more data received")
            break
        
        # Decode the raw JSON data
        raw_json = data.decode('utf-8')
        print("Raw JSON Data:", raw_json)  # Print raw JSON string for debugging
        
        # Parse JSON data
        try:
            parsed_data = json.loads(raw_json)  # Parse JSON string into a Python dictionary
            print("Parsed JSON Data:", parsed_data)  # Print the parsed JSON object
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    # Close the connection and socket
    conn.close()
    receiver_socket.close()
    print("Socket closed.")
