import numpy as np
from dataclasses import dataclass
import space


@dataclass
class Ball:
    r: int
    z: int
    potential: float
    radius: int


BALL = Ball(r=0, z=4, radius=2, potential=10)

FAR_POTENTIAL = 5


def set_boundary(current_space: space.Space):
    _set_ball_boundary(current_space)
    _set_far_boundary(current_space)
    _set_z_zero_boundary(current_space)


def _set_ball_boundary(current_space):
    for r, z in current_space.get_changeable():
        distance_from_ball = np.sqrt((z - 4) ** 2 + r ** 2)
        if distance_from_ball <= BALL.radius:
            current_space.set_point(
                r=np.round(r, decimals=current_space.decimals),
                z=np.round(z, decimals=current_space.decimals),
                val=BALL.potential,
                is_changeable=False,
            )


def _set_far_boundary(current_space):
    for r in np.arange(0, current_space.rmax + current_space.h, current_space.h):
        current_space.set_point(r, current_space.zmax, 0, is_changeable=False)

    for z in np.arange(current_space.h, current_space.zmax, current_space.h):
        current_space.set_point(current_space.rmax, z, 0, is_changeable=False)


def _set_z_zero_boundary(current_space):
    for r in np.arange(0, current_space.rmax + current_space.h, current_space.h):
        current_space.set_point(r, 0, 0, is_changeable=False)
