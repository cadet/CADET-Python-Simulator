import pytest
import numpy as np
import xarray as xr
from CADETPythonSimulator.state import State
from CADETPythonSimulator.geometry import (
    Cylinder,
    Sphere,
    DiscretizedGeometry,
    RadialAngularSymmetry,
    SphericalSymmetry,
    WellMixed,
    AngularSymmetry,
)


# %% Tests for geometry-native State interface


class TestStateGeometryInterface:
    """Tests for State constructed with an explicit DiscretizedGeometry."""

    def _make_1d(self, n_axial=10):
        g = DiscretizedGeometry(Cylinder(length=0.1), RadialAngularSymmetry(), {"axial": n_axial})
        return State(name="bulk", entries={"c": 3, "V": 1}, geometry=g)

    def _make_0d(self):
        g = DiscretizedGeometry(Cylinder(length=1.0), WellMixed())
        return State(name="tank", entries={"c": 2, "V": 1}, geometry=g)

    def _make_2d(self, n_axial=5, n_radial=3):
        g = DiscretizedGeometry(
            Cylinder(length=0.1, radius=0.005),
            AngularSymmetry(),
            {"axial": n_axial, "radial": n_radial},
        )
        return State(name="bulk2d", entries={"c": 2}, geometry=g)

    # ── shape / dof ────────────────────────────────────────────────────────

    def test_1d_shape(self):
        s = self._make_1d()
        assert s.dimension_shape == (10,)
        assert s.n_dimensions == 1
        assert s.n_cells == 10
        assert s.n_entries == 4
        assert s.shape == (10, 4)
        assert s.n_dof == 40

    def test_0d_shape(self):
        s = self._make_0d()
        assert s.dimension_shape == ()
        assert s.n_dimensions == 0
        assert s.n_cells == 1
        assert s.shape == (3,)
        assert s.n_dof == 3

    def test_2d_shape(self):
        s = self._make_2d()
        assert s.dimension_shape == (5, 3)
        assert s.n_dimensions == 2
        assert s.shape == (5, 3, 2)
        assert s.n_dof == 30

    # ── entry access ───────────────────────────────────────────────────────

    def test_getitem_returns_view(self):
        state = self._make_1d()
        c_view = state["c"]
        assert c_view.shape == (10, 3)
        state._s[0, 0] = 99.0
        assert c_view[0, 0] == 99.0

    def test_setitem_mutates_inplace(self):
        state = self._make_1d()
        buf = state._s
        state["c"] = np.ones((10, 3))
        assert state._s is buf
        np.testing.assert_array_equal(state["c"], np.ones((10, 3)))

    def test_getitem_invalid_key(self):
        state = self._make_1d()
        with pytest.raises(KeyError):
            _ = state["nonexistent"]

    # ── in-place mutation invariant ────────────────────────────────────────

    def test_s_setter_inplace(self):
        state = self._make_1d()
        buf = state._s
        state.s = np.ones((10, 4))
        assert state._s is buf
        np.testing.assert_array_equal(state._s, np.ones((10, 4)))

    def test_s_flat_setter_inplace(self):
        state = self._make_1d()
        buf = state._s
        flat = np.arange(40, dtype=float)
        state.s_flat = flat
        assert state._s is buf
        np.testing.assert_array_equal(state.s_flat, flat)

    def test_s_split_setter_inplace(self):
        state = self._make_1d()
        buf = state._s
        state.s_split = {"c": np.ones((10, 3)), "V": np.full((10, 1), 2.0)}
        assert state._s is buf
        np.testing.assert_array_equal(state["c"], np.ones((10, 3)))
        np.testing.assert_array_equal(state["V"], np.full((10, 1), 2.0))

    # ── field (zero-copy xarray views) ────────────────────────────────────

    def test_field_keys(self):
        state = self._make_1d()
        assert set(state.field.keys()) == {"c", "V"}

    def test_field_types(self):
        state = self._make_1d()
        for da in state.field.values():
            assert isinstance(da, xr.DataArray)

    def test_field_c_has_component_dim(self):
        state = self._make_1d()
        da = state.field["c"]
        assert "component" in da.dims
        assert da.shape == (10, 3)

    def test_field_V_no_component_dim(self):
        state = self._make_1d()
        da = state.field["V"]
        assert "component" not in da.dims
        assert da.shape == (10,)

    def test_field_is_zero_copy(self):
        """Mutations to s must be immediately visible in field."""
        state = self._make_1d()
        state["c"] = np.ones((10, 3)) * 5.0
        np.testing.assert_array_equal(state.field["c"].values, np.ones((10, 3)) * 5.0)

    def test_field_persists_across_s_setter(self):
        """field DataArrays remain valid after s is updated in-place."""
        state = self._make_1d()
        da_c = state.field["c"]
        state.s = np.arange(40, dtype=float).reshape(10, 4)
        # da_c still wraps the same buffer — should reflect new values
        np.testing.assert_array_equal(da_c.values, state["c"])

    def test_field_has_axial_coords(self):
        state = self._make_1d()
        assert "axial" in state.field["c"].coords

    def test_2d_field_dims(self):
        state = self._make_2d()
        da = state.field["c"]
        assert "axial" in da.dims
        assert "radial" in da.dims
        assert "component" in da.dims

    # ── port access ────────────────────────────────────────────────────────

    def test_1d_get_port_state_inlet(self):
        state = self._make_1d()
        state.s = np.arange(40, dtype=float).reshape(10, 4)
        port = state.get_port_state("inlet")
        np.testing.assert_array_equal(port["c"], state._s[0, :3])
        np.testing.assert_array_equal(port["V"], state._s[0, 3:4])

    def test_1d_get_port_state_outlet(self):
        state = self._make_1d()
        state.s = np.arange(40, dtype=float).reshape(10, 4)
        port = state.get_port_state("outlet")
        np.testing.assert_array_equal(port["c"], state._s[-1, :3])

    def test_1d_set_port_state_inlet(self):
        state = self._make_1d()
        state.set_port_state("inlet", {"c": np.array([1.0, 2.0, 3.0]), "V": np.array([4.0])})
        np.testing.assert_array_equal(state._s[0, :3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(state._s[0, 3:4], [4.0])

    def test_0d_get_port_state_inlet(self):
        state = self._make_0d()
        state.s = np.array([1.0, 2.0, 3.0])
        port = state.get_port_state("inlet")
        np.testing.assert_array_equal(port["c"], [1.0, 2.0])

    def test_geometry_property(self):
        g = DiscretizedGeometry(Cylinder(length=0.1), RadialAngularSymmetry(), {"axial": 5})
        state = State(name="bulk", entries={"c": 1}, geometry=g)
        assert state.geometry is g

    def test_repr(self):
        state = self._make_1d()
        r = repr(state)
        assert "bulk" in r
        assert "State" in r


# %% Test coupling structure
# %% Run tests

if __name__ == "__main__":
    pytest.main(["test_state.py"])
