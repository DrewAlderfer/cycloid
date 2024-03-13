import numpy as np
from sympy import symbols, diff, lambdify, sin, cos, Matrix, sqrt

class cycloid:
    def __init__(self,
                 num_teeth:int=10,
                 num_pins:int=11,
                 err_correction:float=.01,
                 offset_direction=1,
                 base_radius:float=12.5,
                 pin_diameter:float=5,
                 resolution:int=40):

        self.num_teeth = num_teeth
        self.num_pins = num_pins
        self.res = resolution
        self.base_radius = base_radius
        self.pin_dia = pin_diameter
        self.err_corr = err_correction
        self.dir = offset_direction
        self._cycloid_funcs()
        self._calc_params()

    def _cycloid_funcs(self):
        """
        Uses sympy library to derive the symbolic representation of a cycloidal
        curve, its offset, and the derivative of the offset.

        It then outputs numeric versions of these functions that can be used to
        calculate points along curves with respect to each function.

        There is a class method provided to calculate the full set of points for
        the teeth of a cycloidal gear given the parameters passed to construct
        the class object.
        """
        t:int       = symbols('t')
        gN:int      = self.num_teeth        # Number of Teeth on the Cycloidal Gear
        pC:int      = self.num_pins
        i:float     = pC - gN / gN          # Transmission Ratio
        # l           = self.err_corr
        dir         = self.dir
        r1:float    = self.base_radius      # Pitch Radius of Housing Rollers
        d:float     = self.pin_dia          # Diameter of the Pins

        # e = r1 * (1 - l) / pC             # TODO Implement the error correction
        r2 = r1 /  i
        amount = d / 2
        cost, sint = cos(t), sin(t)
        cosgt, singt = cos(gN*t), sin(gN*t)

        A = Matrix([r2*cosgt+r1+r2, r2*singt]) # Offset cycloid curve with radius `r2` # type: ignore
        R = Matrix([[cost, -sint],  # Rotation Matrix # type: ignore
                    [sint, cost]])

        B:Matrix = R * A    # When `t` is a set of radial positions B(t) will be a set of 
                            # points along the cycloidal arc (t_0 -> t_n).
        dB:Matrix = diff(B, t) # type: ignore
        dx = dB[0]
        dy = dB[1]
        dN = Matrix([dx / sqrt(dx**2 + dy**2), dy / sqrt(dx**2 + dy**2)]) # type: ignore
        rM = Matrix([[0, -1],
                     [1, 0]])
        C = B + ((dir * amount)*rM*dN)
        D = diff(C, t) 

        self.cyc_func     = lambdify([t], B)
        self.offset_func  = lambdify([t], C)
        self.offset_prime = lambdify([t], D)

    def _calc_params(self):
        """
        Calculating the basic parameters of the curve functions.
        """
        n_steps = self.num_teeth
        self.arc = 2 * np.pi / n_steps

        self.n = np.linspace(0, n_steps - 1, n_steps)

        a = self.n * self.arc
        res = self.res
        self.t = np.linspace(a, a + self.arc, res).swapaxes(0, 1)


    def _offset_mask(self):
        """
        Calculates the mask for trimming the offset cycloid curve.

        f1 = cycloid function
        f2 = offset cycloid

        f2 needs to be checked for self-intersection and trimmed at the points
        where that happens.

        It does this by rotating each phase of f2' into alignment with one 
        another/centered on the origin point of the base rotation.

        i.e. base_radius*cos(0), base_radius*sin(0)

        This just comes from observing that when the offset curve (f2) intersects
        itself the slope (f2') of the curve approaches zero.

        By rotating each phase of f2' to where the corresponding phase of f2 would
        be centered on the origin I can then perform a simple check to see which 
        points in the phase have y values below 0.
        """
        cosa = np.cos((self.n+1/2)*-self.arc)
        sina = np.sin((self.n+1/2)*-self.arc)

        arc_rotations = np.array([[cosa, -sina],
                                  [sina, cosa]], dtype=np.float32).transpose(2, 0, 1)
        """
        dt_xy comes in with shape (2, 1, Batch, Point)
        It needs to be in shape   (Batch, Point, 2, 1)
        to be multiplied with the rotation matrix
        """
        dt_xy = self.offset_prime(self.t).transpose(2, 3, 0, 1)
        rot_matrix = np.repeat(np.expand_dims(arc_rotations, axis=1), self.res, axis=1)
        dt_xy_r = np.matmul(rot_matrix, dt_xy) # multiply the matrix by the point vectors
        """
        ** making the mask **

        1. Select only the `y` values from `dt_xy_r`
        2. Create the mask with np.less (`True` for any value under 0)
        3. Repeat each value at the -2 axis: 

            (Batch, Point, 1, 1) -> (Batch, Point, 2, 1)

        return the mask
        """
        return np.repeat(np.less(dt_xy_r[:,:,-1:,:],0), 2, axis=2).squeeze()

    def get_points(self):
        """
        Calculates and returns the points of the cycloid curve.

        Returns:
            Tuple[NDArray, NDArray]
        """
        trimmed_offset_xy = np.ma.masked_array(self.offset_func(self.t), mask=self._offset_mask())
        return self.cyc_func(self.t), trimmed_offset_xy
