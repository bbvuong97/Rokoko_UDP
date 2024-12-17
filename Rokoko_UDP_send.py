import socket
import json
import numpy as np
import time

# Define the hand joint names for left and right hands
left_hand_joint_names = [
    "leftHand", 'leftThumbProximal', 'leftThumbMedial', 'leftThumbDistal', 'leftThumbTip',
    'leftIndexProximal', 'leftIndexMedial', 'leftIndexDistal', 'leftIndexTip',
    'leftMiddleProximal', 'leftMiddleMedial', 'leftMiddleDistal', 'leftMiddleTip',
    'leftRingProximal', 'leftRingMedial', 'leftRingDistal', 'leftRingTip',
    'leftLittleProximal', 'leftLittleMedial', 'leftLittleDistal', 'leftLittleTip'
]

right_hand_joint_names = [
    "rightHand", 'rightThumbProximal', 'rightThumbMedial', 'rightThumbDistal', 'rightThumbTip',
    'rightIndexProximal', 'rightIndexMedial', 'rightIndexDistal', 'rightIndexTip',
    'rightMiddleProximal', 'rightMiddleMedial', 'rightMiddleDistal', 'rightMiddleTip',
    'rightRingProximal', 'rightRingMedial', 'rightRingDistal', 'rightRingTip',
    'rightLittleProximal', 'rightLittleMedial', 'rightLittleDistal', 'rightLittleTip'
]

def normalize_positions(hand_positions, middle_proximal_idx):
    """
    Normalize hand joint positions relative to the middle proximal joint and scale by bone length.
    """
    wrist_position = hand_positions[0]
    middle_proximal_position = hand_positions[middle_proximal_idx]
    bone_length = np.linalg.norm(wrist_position - middle_proximal_position)
    normalized_positions = (middle_proximal_position - hand_positions) / bone_length
    return normalized_positions

def forward_data(data, destination_ip, destination_port):
    """
    Sends the processed data to the specified IP address and port using UDP.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(data, (destination_ip, destination_port))

def start_server(local_ip, local_port, ubuntu_ip, ubuntu_port):
    receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver_socket.bind((local_ip, local_port))
    print(f"Listening for UDP data on {local_ip}:{local_port}...")

    try:
        incomplete_data = ""
        while True:
            # Increase buffer size for larger data
            data, addr = receiver_socket.recvfrom(65535)  # Larger buffer size to handle bigger payloads
            print(f"Data received from {addr}")

            # Accumulate incoming data
            incomplete_data += data.decode('utf-8')

            try:
                # Try to parse JSON from the accumulated data
                parsed_json = json.loads(incomplete_data)

                # Debug the parsed JSON structure
                # print("Parsed JSON Structure:", json.dumps(parsed_json, indent=4))  # Pretty print the JSON structure

                # Check for glove availability
                actor = parsed_json["scene"]["actors"][0]
                has_right_glove = actor["meta"]["hasRightGlove"]
                has_left_glove = actor["meta"]["hasLeftGlove"]

                if not has_right_glove and not has_left_glove:
                    print("No glove data available. Skipping processing.")
                    incomplete_data = ""  # Reset for the next data
                    continue

                # Initialize arrays for joint data
                right_hand_positions = np.zeros((21, 3))
                right_hand_orientations = np.zeros((21, 4))
                left_hand_positions = np.zeros((21, 3))
                left_hand_orientations = np.zeros((21, 4))

                # Process right-hand data if available
                if has_right_glove:
                    for joint_name in right_hand_joint_names:
                        if joint_name in actor["body"]:
                            joint_data = actor["body"][joint_name]
                            joint_position = np.array(list(joint_data["position"].values()))
                            joint_rotation = np.array(list(joint_data["rotation"].values()))
                            right_hand_positions[right_hand_joint_names.index(joint_name)] = joint_position
                            right_hand_orientations[right_hand_joint_names.index(joint_name)] = joint_rotation
                        else:
                            print(f"Warning: Right joint {joint_name} is missing in the JSON data. Skipping...")

                # Process left-hand data if available
                if has_left_glove:
                    for joint_name in left_hand_joint_names:
                        if joint_name in actor["body"]:
                            joint_data = actor["body"][joint_name]
                            joint_position = np.array(list(joint_data["position"].values()))
                            joint_rotation = np.array(list(joint_data["rotation"].values()))
                            left_hand_positions[left_hand_joint_names.index(joint_name)] = joint_position
                            left_hand_orientations[left_hand_joint_names.index(joint_name)] = joint_rotation
                        else:
                            print(f"Warning: Left joint {joint_name} is missing in the JSON data. Skipping...")

                # Normalize positions relative to the middle proximal joint
                data_to_forward = {}

                if has_right_glove:
                    middle_proximal_idx = right_hand_joint_names.index('rightMiddleProximal')
                    normalized_right_positions = normalize_positions(right_hand_positions, middle_proximal_idx)
                    data_to_forward["normalizedRightHandPositions"] = normalized_right_positions.tolist()
                    data_to_forward["rawRightHandPositions"] = right_hand_positions.tolist()
                    data_to_forward["rightHandOrientations"] = right_hand_orientations.tolist()

                if has_left_glove:
                    middle_proximal_idx = left_hand_joint_names.index('leftMiddleProximal')
                    normalized_left_positions = normalize_positions(left_hand_positions, middle_proximal_idx)
                    data_to_forward["normalizedLeftHandPositions"] = normalized_left_positions.tolist()
                    data_to_forward["rawLeftHandPositions"] = left_hand_positions.tolist()
                    data_to_forward["leftHandOrientations"] = left_hand_orientations.tolist()

                # Send the processed data
                forward_data(json.dumps(data_to_forward).encode(), ubuntu_ip, ubuntu_port)
                print(f"Data forwarded to {ubuntu_ip}:{ubuntu_port}")

                # Reset the accumulator after successful parsing
                incomplete_data = ""

            except json.JSONDecodeError:
                # If parsing fails, wait for more data
                print("Waiting for more data to complete JSON...")

    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        receiver_socket.close()
        print("Socket closed.")

if __name__ == "__main__":
    local_ip = "127.0.0.1"
    local_port = 14042
    ubuntu_ip = "10.136.92.174"  # Replace with the receiver's IP
    ubuntu_port = 15000          # Replace with the receiver's port
    start_server(local_ip, local_port, ubuntu_ip, ubuntu_port)

