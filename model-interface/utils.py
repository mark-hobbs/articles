import os
import numpy as np

import pypd

mm_to_m = 1e-3
m_to_mm = 1e3


def load_data_file(filename="half-notched-beam.csv"):
    """
    Determine the location of the example and construct the path to the data
    file dynamically.
    """
    file_path = os.path.join(os.getcwd(), "data", filename)
    return np.genfromtxt(file_path, delimiter=",")


def objective(model):
    """
    Mean Square Error (MSE)
    """
    data_file = load_data_file()

    cmod = data_file[:, 0]
    load_min = data_file[:, 1]
    load_max = data_file[:, 2]
    load_mean = (load_max + load_min) / 2

    load_model = -np.array(model.penetrators[0].penetrator_force_history) * n_div_z
    cmod_model = np.array(model.observations[1].history) - np.array(
        model.observations[0].history
    )

    load_model_interp = np.interp(cmod, cmod_model[:, 0] * m_to_mm, load_model[:, 1])
    return np.mean(np.nan_to_num(load_model_interp - load_mean, nan=0) ** 2)


def build_particle_coordinates(dx, n_div_x, n_div_y):
    particle_coordinates = np.zeros([n_div_x * n_div_y, 2])
    counter = 0

    for i_y in range(n_div_y):  # Depth
        for i_x in range(n_div_x):  # Length
            coord_x = dx * i_x
            coord_y = dx * i_y
            particle_coordinates[counter, 0] = coord_x
            particle_coordinates[counter, 1] = coord_y
            counter += 1

    return particle_coordinates


def build_boundary_conditions(particles):
    bc_flag = np.zeros((len(particles), 2), dtype=np.intc)
    bc_unit_vector = np.zeros((len(particles), 2), dtype=np.intc)
    return bc_flag, bc_unit_vector


def setup_problem(k, alpha):
    dx = 2.5 * mm_to_m
    length = 175 * mm_to_m
    depth = 50 * mm_to_m
    width = 50 * mm_to_m
    n_div_x = np.rint(length / dx).astype(int)
    n_div_y = np.rint(depth / dx).astype(int)
    n_div_z = np.rint(width / dx).astype(int)
    notch = [
        np.array([(length * 0.5) + (dx * 0.5), 0]),
        np.array([(length * 0.5) + (dx * 0.5), depth * 0.5]),
    ]

    x = build_particle_coordinates(dx, n_div_x, n_div_y)
    flag, unit_vector = build_boundary_conditions(x)  # TODO: not needed

    material = pypd.Material(
        name="quasi-brittle", E=37e9, Gf=143.2, density=2346, ft=3.9e6
    )
    bc = pypd.BoundaryConditions(flag, unit_vector, magnitude=0)
    particles = pypd.ParticleSet(x, dx, bc, material)

    radius = 25 * mm_to_m
    penetrators = []
    penetrators.append(
        pypd.Penetrator(
            np.array([0.5 * length, depth + radius - dx]),
            np.array([0, 1]),
            np.array([0, -0.4 * mm_to_m]),
            radius,
            particles,
            name="Penetrator",
            plot=False,
        )
    )
    penetrators.append(
        pypd.Penetrator(
            np.array([0.5 * depth, -radius]),
            np.array([0, 0]),
            np.array([0, 0]),
            radius,
            particles,
            name="Support - left",
            plot=False,
        )
    )
    penetrators.append(
        pypd.Penetrator(
            np.array([3 * depth, -radius]),
            np.array([0, 0]),
            np.array([0, 0]),
            radius,
            particles,
            name="Support - right",
            plot=False,
        )
    )

    observations = []
    observations.append(
        pypd.Observation(
            np.array([77.5 * mm_to_m, 0]), particles, period=1, name="CMOD - left"
        )
    )
    observations.append(
        pypd.Observation(
            np.array([97.5 * mm_to_m, 0]), particles, period=1, name="CMOD - right"
        )
    )

    bonds = pypd.BondSet(
        particles,
        constitutive_law=pypd.NonLinear,
        constitutive_law_params={"alpha": alpha, "k": k},
        surface_correction=True,
        notch=notch,
    )

    return pypd.Model(particles, bonds, penetrators, observations)
