from typing import Union, Optional, Tuple 

import numpy as np
from numpy.typing import NDArray

def rotation(input, t):
    """
    Takes an NDArray of points and a vector of rotations to apply to those
    points.

    Args:
        t: A vector containing rotation angles for a set of points.
        input: An array of points with the shape (t.size, 2, 1)

    Returns:
        An NDArray of points with the applied rotation.

    Rasies:
        ValueError: t must be a 1D array.
        ValueError: input must have shape (n,...,2, 1)
    """
    if len(t.shape) > 1:
        raise ValueError('`t` must be a 1D array')
    if input.shape[-1] > 1:
        raise ValueError('input must have shape (n,..., 2, 1)')

    cos, sin = (np.array([np.cos(t)], dtype=np.float32).swapaxes(0, 1),
                np.array([np.sin(t)], dtype=np.float32).swapaxes(0, 1))

    print(f"cos: {cos.shape}\nsin: {sin.shape}")

    R = np.stack([np.concatenate([cos, -sin], axis=-1),
                    np.concatenate([sin, cos], axis=-1)], axis=-2)
    
    if R.shape[-3] != input.shape[-3]:
        R = np.full(input.shape[:-2] + R.shape[-2:], np.expand_dims(R, axis=1), dtype=np.float32)
    print(f"R: {R.shape}\ninput: {input.shape}")

    return np.matmul(R,  input)

# Base Circle
class BaseCircle:
    def __init__(self, size:float=12.5, resolution:int=50) -> None:
        self.points = np.linspace(0, np.pi * 2, num=resolution, dtype=np.float32)
        self.radius = size
        self.x, self.y = np.array([size * np.cos(self.points), size * np.sin(self.points)], dtype=np.float32)

class Pins:
    def __init__(self,
                 dia:float,
                 base_radius:float=10,
                 pin_count:int=11,
                 resolution:float=25):

        two_pi = np.pi * 2
        size = dia / 2
        arc_length = (two_pi) / pin_count
        t = np.arange(0, two_pi, two_pi / resolution, dtype=np.float32)

        pin_rad = np.arange(0, stop=two_pi, step=arc_length, dtype=np.float32)
        offset = base_radius + size

        pin_pos = np.array([offset * np.cos(pin_rad), offset * np.sin(pin_rad)], dtype=np.float32).swapaxes(0, 1).reshape(11, 2, 1)
        pin_pos = np.expand_dims(pin_pos, axis=1)

        pin = np.array([size * np.cos(t), size * np.sin(t)], dtype=np.float32).swapaxes(0, 1).reshape(25, 2, 1)
        self.pins = np.repeat(np.expand_dims(pin, axis=0), pin_count, axis=0) + pin_pos
        print(self.pins.shape)

        return None

class Cycloid:
    def __init__(self,
                 base_radius:float=10,
                 reduction:float=.1,
                 lobe_count:int=10,
                 resolution:int=10,
                 curve:int=100):

        self.lobe_count = lobe_count
        self.pitch_radius = base_radius
        self.lobe_radius = base_radius * reduction
        self.resolution = resolution
        self.curve = curve

        # Set up the function step size
        # This uses the `np.logspace` function to form the step basis.
        start = 0
        stop = 1
        points = np.logspace(start,
                             stop,
                             num=self.resolution,
                             endpoint=True,
                             base=self.curve)
        # Scale the step size to radians equal to 1/2 the size of a single cycloid
        # lobe. Because the logspace returns an increasing step size and curve
        # starts at the highest step size (ideally) you just substract each step
        # from the desired endpoint. This lets you 'flip' the order of the steps.
        self.t = (np.pi / lobe_count) - ((points * np.pi) / (np.max(points) * lobe_count))
        t2 = (np.pi / lobe_count) + ((points * np.pi) / (np.max(points) * lobe_count))
        self.t = np.concatenate([np.flip(t2), self.t], axis=0)
        # Lobe Rotation Angles
        l_radians = lobe_count * self.t
        # Calculate the lobe curve
        lobe_curve = np.array([[np.cos(l_radians)], [np.sin(l_radians)]], dtype=np.float32).transpose(2, 0, 1)
        self.lobe = self.lobe_radius * lobe_curve
        # Translate lobe curve to circumference of the base circle
        self.lobe = self.lobe + np.array([[self.lobe_radius + self.pitch_radius], [0]], dtype=np.float32)

        # Rotate the curve around the base circle 
        self.curve_points = rotation(self.lobe, self.t)
        # Make a copy of the curve for each lobe in the cycloid
        self.points = np.repeat(np.expand_dims(self.curve_points, axis=0), self.lobe_count, axis=0)
        # Create a 1D array of the rotations around the base circle.
        base_rotation = np.arange(0, 2 * np.pi, step=2*np.pi/lobe_count, dtype=np.float32)
        # Rotate the copies of the curve so that they form the complete cycloid
        self.points = rotation(self.points, base_rotation)



