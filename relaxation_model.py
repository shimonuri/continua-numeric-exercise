import logging
import pickle
import copy
import numpy as np

import space
import boundary_setter

logging.basicConfig(level=logging.INFO)

BALL_CHARGE = 20


class RelaxationModel:
    def __init__(self, rmax, zmax, h, max_diff):
        self._previous_space = None
        self.space = space.Space(rmax, zmax, h)
        self._max_diff = max_diff
        self._is_initialized = False

    def initDatabase(self):
        boundary_setter.set_boundary(self.space)
        self._set_initial_values()
        self._is_initialized = True

    def relax(self, pickle_path):
        if not self._is_initialized:
            logging.error("Not initialized")
            return

        logging.info("Relaxation Model Begins")
        iteration_number = 0
        while not self._is_finished(self._max_diff):
            logging.info(f"Iteration {iteration_number} Begins")
            logging.info(f"(i.{iteration_number}) Saving Space Copy")
            self._save_copy()
            logging.info(f"(i.{iteration_number}) Updating Space")
            self._update_space(self.space.h)
            logging.info(f"Iteration {iteration_number} Finished")
            iteration_number += 1

        logging.info(f"Relaxation Model is Finished")
        logging.info(f"Dumping space into pickle ({pickle_path})")
        self.space.create_map()
        self._dump_space(pickle_path)
        self.space.plot_r()

    def _dump_space(self, pickle_path):
        with open(pickle_path, "wb") as fd:
            pickle.dump(self.space, fd)

    def _save_copy(self):
        del self._previous_space
        self._previous_space = copy.deepcopy(self.space)

    def _is_finished(self, max_diff):
        if self._previous_space is None:
            return False

        actual_max_diff = 0
        for r, z in self.space.get_changeable():
            old_value = self._previous_space.get_point(r, z)
            new_value = self.space.get_point(r, z)
            if old_value != 0:
                diff = (new_value - old_value) / old_value
            else:
                if new_value != 0:
                    diff = 100
                else:
                    diff = 0
            if diff > actual_max_diff:
                actual_max_diff = diff

        logging.info(f"Current diff is {actual_max_diff}")
        if actual_max_diff <= max_diff:
            return True

        return False

    def _update_space(self, h):
        for i, (r, z) in enumerate(self.space.get_changeable()):
            self.space.set_point(
                r, z, self._get_new_potential(h, r, z), is_changeable=True
            )

    def _set_initial_values(self):
        for i, (r, z) in enumerate(self.space.get_changeable()):
            self.space.set_point(
                r, z, self._get_initial_potential(r, z), is_changeable=True
            )

    def _get_initial_potential(self, r, z):
        distance_ball_1 = np.sqrt((z - 4) ** 2 + r ** 2)
        distance_ball_2 = np.sqrt((z + 4) ** 2 + r ** 2)
        return BALL_CHARGE / distance_ball_1 - BALL_CHARGE / distance_ball_2

    def _get_new_potential(self, h, r, z):
        if r + h <= self.space.rmax:
            r_plus = self.space.get_point(r + h, z)
        else:
            r_plus = 0
        if r - h >= 0:
            r_minus = self.space.get_point(r - h, z)
        else:
            r_minus = 0
        if z + h <= self.space.zmax:
            z_plus = self.space.get_point(r, z + h)
        else:
            z_plus = 0
        if z - h >= 0:
            z_minus = self.space.get_point(r, z - h)
        else:
            z_minus = 0

        if r != 0:
            r_delta = h / (2 * r)
        else:
            r_delta = 0

        new_potential = r_plus * (1 + r_delta)
        new_potential += r_minus * (1 - r_delta)
        new_potential += z_plus + z_minus
        new_potential *= 1 / 4
        return new_potential
