import socket

# Settings
local_ip = "127.0.0.1"  # IP Rokoko is streaming to (your local machine)
local_port = 14043       # Port Rokoko is streaming to
remote_ip = "192.168.1.100"  # Replace with the receiving computer's IP
remote_port = 15000      # Port to send data to on the receiving computer

# Create a UDP socket for forwarding
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Create a TCP or UDP socket to receive Rokoko data
receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
receiver_socket.bind((local_ip, local_port))
receiver_socket.listen(1)

print(f"Listening for data from Rokoko Studio on {local_ip}:{local_port}...")

# Accept Rokoko connection
conn, addr = receiver_socket.accept()
print(f"Connection established with {addr}")

try:
    while True:
        # Receive data from Rokoko Studio
        data = conn.recv(4096)  # Buffer size
        if not data:
            break
        
        # Print the data in real-time
        print(f"Received: {data.decode('utf-8')}")

        # Send the data over UDP to the remote computer
        udp_socket.sendto(data, (remote_ip, remote_port))
        print(f"Forwarded data to {remote_ip}:{remote_port}")
except KeyboardInterrupt:
    print("Stopped by user")
finally:
    conn.close()
    receiver_socket.close()
    udp_socket.close()
