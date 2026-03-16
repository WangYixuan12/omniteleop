import time
import numpy as np
import viser
from scipy.spatial.transform import Rotation
from omniteleop.leader.communication.webxr_vr_reader import WebXRVRReader


def mat_to_pos_wxyz(mat: np.ndarray):
    """Extract position and (w,x,y,z) quaternion from a 4×4 SE(3) matrix."""
    pos = mat[:3, 3]
    xyzw = Rotation.from_matrix(mat[:3, :3]).as_quat()  # scipy: x,y,z,w
    wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
    return pos, wxyz


reader = WebXRVRReader(port=5067, ssl_certfile="/home/yixuan/omniteleop/tests/cert.pem", ssl_keyfile="/home/yixuan/omniteleop/tests/key.pem")
reader.start()

server = viser.ViserServer(host="0.0.0.0", port=8080)
print("Viser running at http://localhost:8080")

print("Waiting for first VR frame …")
reader.wait_for_data()
print("Streaming — open the viser URL in a browser to visualise.")

while True:
    data = reader.get_latest_transformation()
    print(f"head pose {data["head"]}")
    print(f"left wrist pose {data["left_wrist"]}")
    print(f"right wrist pose {data["right_wrist"]}")
    print(f"right index trigger {data["right_index_trigger"]}")
    print(f"right hand trigger {data["right_hand_trigger"]}")
    if data is None:
        time.sleep(0.01)
        continue

    for name, key in [("head", "head"), ("left_wrist", "left_wrist"), ("right_wrist", "right_wrist")]:
        mat = data[key]
        pos, wxyz = mat_to_pos_wxyz(mat)
        server.scene.add_frame(
            name,
            wxyz=wxyz,
            position=pos,
            axes_length=0.1,
            axes_radius=0.005,
        )

    time.sleep(0.01)
