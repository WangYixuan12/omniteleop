"""Surface distance between convex collision volumes (not vertex distances)."""
import coal
import numpy as np


def hull(vertices):
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
        raise ValueError('Convex hull requires finite Nx3 vertices')
    points = coal.StdVec_Vec3s()
    for point in vertices:
        points.append(point)
    return coal.Convex.convexHull(points, True, 'Qt')


def distance(a, Ta, b, Tb):
    result = coal.DistanceResult()
    value = float(coal.distance(a, Ta, b, Tb, coal.DistanceRequest(), result))
    if not np.isfinite(value):
        raise ValueError('Nonfinite collision distance')
    return value
