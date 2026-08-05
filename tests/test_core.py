import numpy as np

from cmc_remodelling.core import (
    D_POST,
    D_PRE,
    NestedComplexState,
    birth_commutation_residual,
    necrotic_birth_maps,
    orientation_invariance_residual,
    project_nested,
    radii_to_shell_capacities,
    shell_capacities_to_radii,
)


def test_roundtrip_shell_capacities():
    radii = np.array([[300.0, 220.0, 100.0], [100.0, 0.0, 0.0]])
    rebuilt = shell_capacities_to_radii(radii_to_shell_capacities(radii))
    assert np.allclose(rebuilt, radii)


def test_projection_is_nested():
    projected = project_nested(np.array([[10.0, 20.0, 30.0]]))
    assert 0 <= projected[0, 2] <= projected[0, 1] <= projected[0, 0]


def test_necrotic_birth_is_compatible():
    p_cells, p_interfaces = necrotic_birth_maps()
    assert p_cells.shape == (3, 2)
    assert p_interfaces.shape == (2, 1)
    assert np.allclose(D_POST @ p_cells, p_interfaces @ D_PRE)
    assert birth_commutation_residual() == 0.0


def test_orientation_audit():
    assert orientation_invariance_residual() == 0.0


def test_active_complex_changes_at_necrosis():
    pre = NestedComplexState(200.0, 120.0, 0.0)
    post = NestedComplexState(260.0, 190.0, 60.0)
    assert pre.active_incidence().shape == (1, 2)
    assert post.active_incidence().shape == (2, 3)
