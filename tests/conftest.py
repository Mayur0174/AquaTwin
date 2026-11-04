import numpy as np
import pytest

from aquatwin.config import AquaTwinConfig
from aquatwin.network.generator import generate_network
from aquatwin.physics.degradation import hazen_williams_c
from aquatwin.sensors.registry import place_sensors
from aquatwin.truth.builder import ModelBuilder


@pytest.fixture(scope="session")
def cfg():
    return AquaTwinConfig()


@pytest.fixture(scope="session")
def network(cfg):
    return generate_network(cfg.network)


@pytest.fixture(scope="session")
def sensors(network, cfg):
    return place_sensors(network, cfg.sensors)


@pytest.fixture(scope="session")
def base(network):
    """(builder, nominal demand m3/s, tank levels, HW C) for direct hydraulic experiments."""
    b = ModelBuilder(network)
    n, p = network.nodes, network.pipes
    dem = n.base_demand_ls.fillna(0).to_numpy() / 1000.0
    hw = hazen_williams_c(p.material.to_numpy(), p.age_years.to_numpy(float))
    levels = n.loc[b.tank_ids, "init_level"].to_numpy(float)
    return b, dem, levels, hw


@pytest.fixture(scope="session")
def junction_mask(network):
    return network.nodes.node_type.to_numpy() == "junction"


def pipe_far_from_tanks(network):
    p = network.pipes
    D = network.distance_matrix
    dist = [min(D.at[r.start_node, t] for t in network.tank_ids) for r in p.itertuples()]
    return p.index[int(np.argmax(dist))]
