"""Tiny length-prefixed pickle wire protocol for the WBC bridge.

Pure stdlib so BOTH the dexmate WBC service (py3.12) and the OmniGibson behavior
client (py3.11) can import it. Payloads are plain Python floats/lists (numpy arrays
converted to .tolist() on the wire) so there is no cross-numpy-version pickle risk.
"""
import pickle
import struct

_PROTOCOL = 4  # supported by both py3.11 and py3.12


def send_msg(sock, obj) -> None:
    data = pickle.dumps(obj, protocol=_PROTOCOL)
    sock.sendall(struct.pack(">I", len(data)) + data)


def _recv_all(sock, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed by peer")
        buf.extend(chunk)
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack(">I", _recv_all(sock, 4))
    return pickle.loads(_recv_all(sock, n))
