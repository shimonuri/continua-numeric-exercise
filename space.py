import matplotlib.pyplot as plt
import numpy as np


class Space(object):
    def __init__(self, rmax, zmax, h):
        self.rmax = rmax
        self.zmax = zmax
        self.h = h
        self.decimals = len(str(h).split(".")[-1])
        self.space = {}
        self.__changeable = []
        self.__set_space()

    def set_point(self, r, z, val, is_changeable=True):
        r = np.round(r, decimals=self.decimals)
        z = np.round(z, decimals=self.decimals)
        # Check if the point is changeable:
        if (r, z) in self.space:
            if not self.space[(r, z)]["is_changeable"]:
                raise Exception(f"The point {(r,z)} is not changeable!")

        self.space[(r, z)] = {"potential": val, "is_changeable": is_changeable}
        if is_changeable:
            self.__changeable.append((r, z))
        elif (r, z) in self.__changeable:
            self.__changeable.remove((r, z))

    def get_point(self, r, z):
        r = np.round(r, decimals=self.decimals)
        z = np.round(z, decimals=self.decimals)
        if (r, z) not in self.space:
            raise KeyError(f"{(r, z)} is not in the space")

        return self.space[(r, z)]["potential"]

    def __set_space(self):
        for i in np.arange(0, self.rmax + self.h, self.h):
            for j in np.arange(0, self.zmax + self.h, self.h):
                self.set_point(i, j, 8)

    def get_changeable(self):
        return sorted(self.__changeable)

    def _convert_space_into_matrix(self):
        matrix = [
            [0 for i in np.arange(0, self.rmax + self.h, self.h)]
            for i in np.arange(0, self.zmax + self.h, self.h)
        ]
        for r, z in self.space:
            r_i = int(np.round(r / self.h))
            z_i = int(np.round(z / self.h))
            matrix[z_i][r_i] = self.get_point(r, z)

        return matrix

    def create_map(self):
        matrix = self._convert_space_into_matrix()
        fig, ax = plt.subplots(1, 1)
        img = plt.imshow(matrix, extent=[0, self.rmax, self.zmax, 0])
        fig.colorbar(img)
        plt.xlabel("r")
        plt.ylabel("z")
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        # ax.set_xlim(ax.get_xlim()[::-1])
        plt.show()

    def plot_r(self, r=0):
        z_vals = np.arange(0, self.zmax + self.h, self.h)
        potential_vals = [self.get_point(r, z) for z in z_vals]
        plt.plot(z_vals, potential_vals)
        plt.xlabel("z")
        plt.ylabel("Potential")
        plt.show()


#
# if __name__ == '__main__':
#     s = Space(rmax=8, zmax=14, h=0.1)
#     s.set_point(2, 2, 2, is_changeable=False)
#     # s.set_point(2, 2, 3)
#     print(s.get_point(2, 2))
#     s.create_map()
