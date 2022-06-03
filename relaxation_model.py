import numpy as np
import matplotlib.pyplot as plt
import dataclasses

BOX_SIZE = 50


@dataclasses.dataclass
class Square:
    x_pos: int
    y_pos: int
    size: int


class Model:
    def __init__(
        self, size, max_diff=0.0000001, start_value=1, max_iter=10000, squares=None
    ):
        if not squares:
            squares = []

        self._size = size
        self._current_potential = np.ones((size, size)) * start_value
        self._delta_x = 1 / size
        self._impose_boundary(self._current_potential)
        self._set_init_values(self._current_potential)
        self._squares = squares
        self._put_squares(self._current_potential)
        self._max_diff = max_diff
        self._max_iter = max_iter

    @property
    def current_potential(self):
        return self._current_potential

    @property
    def max_iter(self):
        return self._max_iter

    @property
    def max_diff(self):
        return self._max_diff

    def solve(self):
        max_diff = np.inf
        iter_amount = 0
        while max_diff > self.max_diff and iter_amount < self.max_iter:
            max_diff = self._make_step()
            iter_amount += 1

        return self.current_potential, self._get_velocity(self.current_potential)

    def _make_step(self):
        new_potential = np.copy(self._current_potential)
        (n, m) = self.current_potential.shape
        new_potential[1 : n - 1, 1 : m - 1] = (
            self.current_potential[0 : n - 2, 1 : m - 1]
            + self.current_potential[2:n, 1 : m - 1]
            + self.current_potential[1 : n - 1, 0 : m - 2]
            + self.current_potential[1 : n - 1, 2:m]
        ) / 4
        self._impose_boundary(new_potential)
        self._put_squares(new_potential)
        max_diff = np.max(np.abs(new_potential - self.current_potential))
        self._current_potential = new_potential
        return max_diff

    def _put_squares(self, potential):
        for square in self._squares:
            self._put_square(potential, square.x_pos, square.y_pos, square.size)

    def _put_square(self, potential, x_pos, y_pos, length):
        potential[y_pos : y_pos + length, x_pos : x_pos + length] = np.zeros(
            (length, length)
        )
        potential[y_pos : y_pos + length, x_pos] = potential[
            y_pos : y_pos + length, x_pos - 1
        ]
        potential[y_pos : y_pos + length, x_pos + length] = potential[
            y_pos : y_pos + length, x_pos + length + 1
        ]
        potential[y_pos, x_pos : x_pos + length] = potential[
            y_pos - 1, x_pos : x_pos + length
        ]
        potential[y_pos + length, x_pos : x_pos + length] = potential[
            y_pos + length + 1, x_pos : x_pos + length
        ]

    def _set_init_values(self, potential):
        (n, m) = potential.shape
        potential[0 : n // 4, 0 : m // 3] = np.zeros((n // 4, m // 3))

    def _impose_boundary(self, potential):
        (n, m) = potential.shape
        # all sides - zero derivative
        potential[:, 0:1] = potential[:, 1:2]
        potential[:, n - 1 : n] = potential[:, n - 2 : n - 1]
        potential[0:1, :] = potential[1:2, :]
        potential[m - 1 : m, :] = potential[m - 2 : m - 1, :]
        # left corner
        potential[n // 4 : 3 * n // 4, 0:1] = (
            potential[n // 4 : 3 * n // 4, 1:2] - self._delta_x * 100
        )
        # right corner
        potential[:, n - 1 : n] = potential[:, n - 2 : n - 1] + self._delta_x * 100 / 2

    def _get_velocity(self, potential):
        x_velocity = self._get_x_velocity(potential)
        y_velocity = self._get_y_velocity(potential)
        return x_velocity, y_velocity

    def _should_skip_print(self, x, y):
        for square in self._squares:
            if x in range(square.x_pos, square.x_pos + square.size + 1) and y in range(
                square.y_pos, square.y_pos + square.size + 1
            ):
                return True

        return False

    def _get_x_velocity(self, potential):
        (n, m) = potential.shape
        x_velocity = np.zeros((n, m))
        for x in range(n):
            for y in range(m):
                if self._should_skip_print(x, y):
                    continue
                if x == 0:
                    x_velocity[y, x] = (
                        potential[y, x + 1] - potential[y, x]
                    ) / self._delta_x
                elif x == n - 1:
                    x_velocity[y, x] = (
                        potential[y, x] - potential[y, x - 1]
                    ) / self._delta_x
                else:
                    x_velocity[y, x] = (potential[y, x + 1] - potential[y, x - 1]) / (
                        2 * self._delta_x
                    )
        return x_velocity

    def _get_y_velocity(self, potential):
        (n, m) = potential.shape
        y_velocity = np.zeros((n, m))
        for x in range(n):
            for y in range(m):
                if self._should_skip_print(x, y):
                    continue
                if y == 0:
                    y_velocity[y, x] = (
                        potential[y + 1, x] - potential[y, x]
                    ) / self._delta_x
                elif y == m - 1:
                    y_velocity[y, x] = (
                        potential[y, x] - potential[y - 1, x]
                    ) / self._delta_x
                else:
                    y_velocity[y, x] = (potential[y + 1, x] - potential[y - 1, x]) / (
                        2 * self._delta_x
                    )
        return y_velocity


def main():
    squares = [Square(x_pos=20, y_pos=35, size=BOX_SIZE // 10)]
    model = Model(BOX_SIZE, squares=squares)
    potential, (x_velocity, y_velocity) = model.solve()
    plt.imshow(potential, cmap="hot", interpolation="nearest")
    plt.colorbar()
    # plt.show()
    _plot_velocity_field(x_velocity, y_velocity)
    _plot_y_axis_kinetic_energy(BOX_SIZE // 2, x_velocity, y_velocity)
    _plot_x_axis_kinetic_energy(BOX_SIZE // 2, x_velocity, y_velocity)


def _plot_y_axis_kinetic_energy(y, x_velocity, y_velocity):
    (n, m) = x_velocity.shape
    kinetic = []
    for x in range(m):
        kinetic.append((1 / 2) * (x_velocity[y, x] ** 2 + y_velocity[y, x] ** 2))

    plt.title("Kinetic Energy versus Y position", fontsize=20)
    plt.xlabel("Y position", fontsize=15)
    plt.ylabel("Kinetic Energy", fontsize=15)
    plt.plot(range(n), kinetic)
    plt.show()


def _plot_x_axis_kinetic_energy(x, x_velocity, y_velocity):
    (n, m) = x_velocity.shape
    kinetic = []
    for y in range(n):
        kinetic.append((1 / 2) * (x_velocity[y, x] ** 2 + y_velocity[y, x] ** 2))

    plt.title("Kinetic Energy versus X position", fontsize=20)
    plt.xlabel("X position", fontsize=15)
    plt.ylabel("Kinetic Energy", fontsize=15)
    plt.plot(range(m), kinetic)
    plt.show()


def _plot_velocity_field(x_velocity, y_velocity):
    x_data = []
    y_data = []
    u_data = []
    v_data = []
    (n, m) = x_velocity.shape
    for x in range(n):
        for y in range(m):
            x_data.append(x)
            y_data.append(y)
            u_data.append(y_velocity[y, x])
            v_data.append(x_velocity[y, x])
    plt.quiver(x_data, y_data, v_data, u_data)
    plt.show()


if __name__ == "__main__":
    main()
