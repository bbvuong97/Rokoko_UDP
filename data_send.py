import socket
import json
import numpy as np
from pybullet_ik_bimanual import LeapPybulletIK
import time
from utils import *
# Define the hand joint names for left and right hands
left_hand_joint_names = ["leftHand",
                         'leftThumbProximal', 'leftThumbMedial', 'leftThumbDistal', 'leftThumbTip',
                         'leftIndexProximal', 'leftIndexMedial', 'leftIndexDistal', 'leftIndexTip',
                         'leftMiddleProximal', 'leftMiddleMedial', 'leftMiddleDistal', 'leftMiddleTip',
                         'leftRingProximal', 'leftRingMedial', 'leftRingDistal', 'leftRingTip',
                         'leftLittleProximal', 'leftLittleMedial', 'leftLittleDistal', 'leftLittleTip']

right_hand_joint_names = ["rightHand",
                          'rightThumbProximal', 'rightThumbMedial', 'rightThumbDistal', 'rightThumbTip',
                          'rightIndexProximal', 'rightIndexMedial', 'rightIndexDistal', 'rightIndexTip',
                          'rightMiddleProximal', 'rightMiddleMedial', 'rightMiddleDistal', 'rightMiddleTip',
                          'rightRingProximal', 'rightRingMedial', 'rightRingDistal', 'rightRingTip',
                          'rightLittleProximal', 'rightLittleMedial', 'rightLittleDistal', 'rightLittleTip']

def normalize_wrt_middle_proximal(hand_positions, is_left=True):
    middle_proximal_idx = left_hand_joint_names.index('leftMiddleProximal')
    if not is_left:
        middle_proximal_idx = right_hand_joint_names.index('rightMiddleProximal')

    wrist_position = hand_positions[0]
    middle_proximal_position = hand_positions[middle_proximal_idx]
    bone_length = np.linalg.norm(wrist_position - middle_proximal_position)
    normalized_hand_positions = (middle_proximal_position - hand_positions) / bone_length
    return normalized_hand_positions

def forward_data(data, destination_ip, destination_port):
    """
    Sends the processed data to the specified IP address and port using UDP.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create a UDP socket
    sock.sendto(data, (destination_ip, destination_port))    # Send data to the specified IP and port

def start_server(port, ubuntu_ip, ubuntu_port, ubuntu_ip_2, ubuntu_port_2):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Using SOCK_DGRAM for UDP
    s.bind(("192.168.0.132", port))
    print(f"Server started, listening on port {port} for UDP packets...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 7777)
    sock.bind(server_address)
    
    pyb = LeapPybulletIK()
    while True:
        start_time = time.time()
        data, address = s.recvfrom(64800)  # Receive UDP packets
        decoded_data = data.decode()
        # print("!!!!")

        _start_time = time.time()
        while True:
            data2, _ = sock.recvfrom(4096)
            if time.time() - _start_time > 0.003:
                break
        decoded_data2 = data2.decode('utf-8')
        # Attempt to parse JSON
        try:
            received_json = json.loads(decoded_data)

            # Initialize arrays to store the positions
            left_hand_positions = np.zeros((21, 3))
            right_hand_positions = np.zeros((21, 3))

            left_hand_orientations = np.zeros((21, 4))
            right_hand_orientations = np.zeros((21, 4))

            # Iterate through the JSON data to extract hand joint positions
            # for joint_name in left_hand_joint_names:
            #     joint_data = received_json["scene"]["actors"][0]["body"][joint_name]
            #     joint_position = np.array(list(joint_data["position"].values()))
            #     joint_rotation = np.array(list(joint_data["rotation"].values()))
            #     left_hand_positions[left_hand_joint_names.index(joint_name)] = joint_position
            #     left_hand_orientations[left_hand_joint_names.index(joint_name)] = joint_rotation

            for joint_name in right_hand_joint_names:
                joint_data = received_json["scene"]["actors"][0]["body"][joint_name]
                joint_position = np.array(list(joint_data["position"].values()))
                joint_rotation = np.array(list(joint_data["rotation"].values()))
                right_hand_positions[right_hand_joint_names.index(joint_name)] = joint_position
                right_hand_orientations[right_hand_joint_names.index(joint_name)] = joint_rotation

            # print(right_hand_positions[0])
            # relative distance to middle proximal joint
            # normalize by bone distance (distance from wrist to middle proximal)   
            # Define the indices of 'middleProximal' in your joint names
            # left_middle_proximal_idx = left_hand_joint_names.index('leftMiddleProximal')
            right_middle_proximal_idx = right_hand_joint_names.index('rightMiddleProximal')

            # Calculate bone length from 'wrist' to 'middleProximal' for both hands
            # left_wrist_position = left_hand_positions[0]
            right_wrist_position = right_hand_positions[0]

            # left_middle_proximal_position = left_hand_positions[left_middle_proximal_idx]
            right_middle_proximal_position = right_hand_positions[right_middle_proximal_idx]

            # left_bone_length = np.linalg.norm(left_wrist_position - left_middle_proximal_position)
            right_bone_length = np.linalg.norm(right_wrist_position - right_middle_proximal_position)

            # Calculate relative positions and normalize
            # normalized_left_hand_positions = (left_middle_proximal_position - left_hand_positions) / left_bone_length
            normalized_right_hand_positions = (right_middle_proximal_position - right_hand_positions) / right_bone_length

            # right_hand_pos = [np.array(pos) for pos in right_hand_positions]
            right_hand_ori = np.array(right_hand_orientations[0])
            # left_hand_pos = [np.array(pos) for pos in left_hand_positions]
            left_hand_ori = np.array(left_hand_orientations[0])

            right_hand_pos = translate_wrist_to_origin(right_hand_positions)
            left_hand_pos = translate_wrist_to_origin(left_hand_positions)
            real_right_robot_hand_q, real_left_robot_hand_q = pyb.compute_IK(right_hand_pos, right_hand_ori, left_hand_pos, left_hand_ori)

            position_data, rotation_data = decoded_data2.split(';')
            position = list(map(float, position_data.split(',')))
            quaternion = list(map(float, rotation_data.split(',')))
            # Prepare the data to be forwarded (convert numpy arrays to bytes)
            data_to_forward = {
                # "leftHandPositions": normalized_left_hand_positions.tolist(),
                "rightHandPositions": normalized_right_hand_positions.tolist(),
                "rawRightHandPositions": right_hand_positions.tolist(),
                # "leftHandPositions": left_hand_positions.tolist(),
                # "rightHandPositions": right_hand_positions.tolist(),
                # "leftHandOrientations": left_hand_orientations.tolist(),
                "rightHandOrientations": right_hand_ori.tolist(),
                "real_right_hand": real_right_robot_hand_q.tolist(),
                # "real_left_hand": real_left_robot_hand_q.tolist(),
                "right_hand_pos_vr": position,
                "right_hand_q_vr": quaternion
            }
            # forward_data(json.dumps(data_to_forward).encode(), ubuntu_ip, ubuntu_port)
            forward_data(json.dumps(data_to_forward).encode(), ubuntu_ip_2, ubuntu_port_2)
            duration = time.time() - start_time

            print("fps:", 1/duration)
            
            # print("\n\n")
            # print("=" * 50)
            # print(np.round(left_hand_positions, 3))
            # print("-" * 50)
            # print(np.round(right_hand_positions, 3))

        except json.JSONDecodeError:
            print("Invalid JSON received:")

if __name__ == "__main__":
    # Update the IP and port to the Ubuntu machine's IP and desired receiving port
    ubuntu_ip = "192.168.0.205"
    ubuntu_port = 14001
    ubuntu_ip_2 = "192.168.0.189"
    ubuntu_port_2 = 14005
    start_server(14551, ubuntu_ip, ubuntu_port, ubuntu_ip_2, ubuntu_port_2)
