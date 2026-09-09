"""Numpy-independent payloads across the behavior / dexmate Python boundary."""
import numpy as np


def pack(value):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("Object arrays cannot cross the simulator bridge")
        return {"__array__": value.tobytes(), "dtype": value.dtype.str, "shape": value.shape}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: pack(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(item) for item in value]
    return value


def unpack(value):
    if isinstance(value, dict):
        if "__array__" in value:
            dtype = np.dtype(value["dtype"])
            if dtype.hasobject:
                raise TypeError("Object arrays cannot cross the simulator bridge")
            return np.frombuffer(value["__array__"], dtype=dtype).reshape(value["shape"]).copy()
        return {key: unpack(item) for key, item in value.items()}
    if isinstance(value, list):
        return [unpack(item) for item in value]
    return value
