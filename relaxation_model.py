import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, size, start_value=0.5):
        self._current_potential = np.random.rand((size, size)) * start_value
        self._impose_boundary()

    @property
    def current_potential(self):
        return self._current_potential

    def solve(self):
        self._make_step()
        return self.current_potential

    def _make_step(self):
        new_potential = np.copy(self._current_potential)
        (n, m) = self.current_potential.shape
        new_potential[1 : n - 1, 1 : m - 1] = (
            self.current_potential[0 : n - 2, 1 : m - 1]
            + self.current_potential[2 : n, 1 : m - 1]
            + self.current_potential[1 : n - 1, 0 : m - 2]
            + self.current_potential[1 : n - 1, 2:m]
        ) / 4
        self._impose_boundary()
        self._current_potential = new_potential

    def _impose_boundary(self):
        pass


def main():
    model = Model(10)
    solution = model.solve()
    plt.imshow(solution, cmap="hot", interpolation="nearest")
    plt.show()


if __name__ == "__main__":
    main()
