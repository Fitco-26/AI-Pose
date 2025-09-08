import numpy as np


def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., shoulder, elbow, wrist)."""
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
        np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def map_angle_to_progress(angle, min_angle=30, max_angle=160):
    """Maps an angle to a 0-100 progress value."""
    angle = max(min_angle, min(max_angle, angle))
    # Invert the progress: 160 degrees is 0% progress, 30 degrees is 100%
    return 100 - int((angle - min_angle) * 100 / (max_angle - min_angle))
