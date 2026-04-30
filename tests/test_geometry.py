import numpy as np
import pytest

from CADETPythonSimulator.geometry import (
    AngularSymmetry,
    AxialAngularSymmetry,
    Cylinder,
    DiscretizedGeometry,
    Frustum,
    HollowCylinder,
    NoSymmetry,
    RadialAngularSymmetry,
    Sphere,
    SphericalSymmetry,
    WellMixed,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def cylinder():
    return Cylinder(length=0.1, radius=0.01)

@pytest.fixture
def hollow_cylinder():
    return HollowCylinder(r_inner=0.005, r_outer=0.02)

@pytest.fixture
def frustum():
    return Frustum(length=0.1, r_inlet=0.01, r_outlet=0.005)

@pytest.fixture
def sphere():
    return Sphere(radius=50e-6)


# ── Shape ──────────────────────────────────────────────────────────────────

class TestShape:
    def test_tank(self, cylinder):
        g = DiscretizedGeometry(cylinder, WellMixed())
        assert g.shape == ()

    def test_column_1d(self, cylinder):
        g = DiscretizedGeometry(cylinder, RadialAngularSymmetry(), {"axial": 20})
        assert g.shape == (20,)

    def test_column_2d(self, cylinder):
        g = DiscretizedGeometry(cylinder, AngularSymmetry(), {"axial": 20, "radial": 5})
        assert g.shape == (20, 5)

    def test_column_3d(self, cylinder):
        g = DiscretizedGeometry(cylinder, NoSymmetry(), {"axial": 20, "radial": 5, "angular": 8})
        assert g.shape == (20, 5, 8)

    def test_radial_column(self, hollow_cylinder):
        g = DiscretizedGeometry(hollow_cylinder, AxialAngularSymmetry(), {"radial": 10})
        assert g.shape == (10,)

    def test_sphere(self, sphere):
        g = DiscretizedGeometry(sphere, SphericalSymmetry(), {"radial": 5})
        assert g.shape == (5,)

    def test_frustum(self, frustum):
        g = DiscretizedGeometry(frustum, RadialAngularSymmetry(), {"axial": 10})
        assert g.shape == (10,)


# ── n_dof ──────────────────────────────────────────────────────────────────

class TestNDof:
    def test_tank(self, cylinder):
        g = DiscretizedGeometry(cylinder, WellMixed())
        assert g.n_dof == 1

    def test_column_1d(self, cylinder):
        g = DiscretizedGeometry(cylinder, RadialAngularSymmetry(), {"axial": 20})
        assert g.n_dof == 20

    def test_column_2d(self, cylinder):
        g = DiscretizedGeometry(cylinder, AngularSymmetry(), {"axial": 20, "radial": 5})
        assert g.n_dof == 100


# ── Coordinates ────────────────────────────────────────────────────────────

class TestCoordinates:
    def test_tank_has_no_coordinates(self, cylinder):
        g = DiscretizedGeometry(cylinder, WellMixed())
        assert g.coordinates == {}

    def test_column_1d_axial_extent(self, cylinder):
        g = DiscretizedGeometry(cylinder, RadialAngularSymmetry(), {"axial": 20})
        z = g.coordinates["axial"]
        assert z[0] == pytest.approx(0.0)
        assert z[-1] == pytest.approx(cylinder.length)
        assert len(z) == 20

    def test_column_2d_coords(self, cylinder):
        g = DiscretizedGeometry(cylinder, AngularSymmetry(), {"axial": 20, "radial": 5})
        assert set(g.coordinates.keys()) == {"axial", "radial"}
        assert g.coordinates["radial"][-1] == pytest.approx(cylinder.radius)

    def test_radial_column_extent(self, hollow_cylinder):
        g = DiscretizedGeometry(hollow_cylinder, AxialAngularSymmetry(), {"radial": 10})
        r = g.coordinates["radial"]
        assert r[0] == pytest.approx(hollow_cylinder.r_inner)
        assert r[-1] == pytest.approx(hollow_cylinder.r_outer)

    def test_sphere_extent(self, sphere):
        g = DiscretizedGeometry(sphere, SphericalSymmetry(), {"radial": 5})
        r = g.coordinates["radial"]
        assert r[0] == pytest.approx(0.0)
        assert r[-1] == pytest.approx(sphere.radius)


# ── External ports ─────────────────────────────────────────────────────────

class TestExternalPorts:
    def test_tank_has_inlet_and_outlet(self, cylinder):
        g = DiscretizedGeometry(cylinder, WellMixed())
        assert "inlet" in g.external_ports
        assert "outlet" in g.external_ports
        assert g.external_ports["inlet"] == ()
        assert g.external_ports["outlet"] == ()

    def test_column_1d_ports(self, cylinder):
        g = DiscretizedGeometry(cylinder, RadialAngularSymmetry(), {"axial": 20})
        assert g.external_ports["inlet"] == (0,)
        assert g.external_ports["outlet"] == (-1,)

    def test_column_2d_ports_span_radial(self, cylinder):
        g = DiscretizedGeometry(cylinder, AngularSymmetry(), {"axial": 20, "radial": 5})
        inlet = g.external_ports["inlet"]
        outlet = g.external_ports["outlet"]
        # inlet/outlet are axial faces — radial dim fully selected
        s = np.zeros(g.shape)
        assert s[inlet].shape == (5,)
        assert s[outlet].shape == (5,)

    def test_radial_column_direction(self, hollow_cylinder):
        g = DiscretizedGeometry(hollow_cylinder, AxialAngularSymmetry(), {"radial": 10})
        # flow: outer (high index) → inner (low index)
        assert g.external_ports["inlet"] == (-1,)
        assert g.external_ports["outlet"] == (0,)

    def test_sphere_has_no_external_ports(self, sphere):
        g = DiscretizedGeometry(sphere, SphericalSymmetry(), {"radial": 5})
        assert g.external_ports == {}


# ── Flux interfaces ────────────────────────────────────────────────────────

class TestFluxInterfaces:
    def test_column_has_no_flux_interfaces(self, cylinder):
        g = DiscretizedGeometry(cylinder, RadialAngularSymmetry(), {"axial": 20})
        assert g.flux_interfaces == {}

    def test_sphere_surface_interface(self, sphere):
        g = DiscretizedGeometry(sphere, SphericalSymmetry(), {"radial": 5})
        assert "flux_interface" in g.flux_interfaces
        assert g.flux_interfaces["flux_interface"] == (-1,)


# ── Spatially varying ──────────────────────────────────────────────────────

class TestSpatiallyVarying:
    def test_cylinder_has_none(self, cylinder):
        g = DiscretizedGeometry(cylinder, RadialAngularSymmetry(), {"axial": 20})
        assert g.spatially_varying == {}

    def test_frustum_has_cross_section(self, frustum):
        g = DiscretizedGeometry(frustum, RadialAngularSymmetry(), {"axial": 10})
        assert "cross_section" in g.spatially_varying
        assert g.spatially_varying["cross_section"].shape == (10,)

    def test_frustum_cross_section_values(self, frustum):
        g = DiscretizedGeometry(frustum, RadialAngularSymmetry(), {"axial": 10})
        A = g.spatially_varying["cross_section"]
        assert A[0] == pytest.approx(np.pi * frustum.r_inlet ** 2)
        assert A[-1] == pytest.approx(np.pi * frustum.r_outlet ** 2)


# ── Error handling ─────────────────────────────────────────────────────────

class TestErrors:
    def test_missing_n_cells_raises(self, cylinder):
        with pytest.raises(ValueError, match="n_cells missing"):
            DiscretizedGeometry(cylinder, RadialAngularSymmetry())

    def test_missing_one_dim_raises(self, cylinder):
        with pytest.raises(ValueError, match="n_cells missing"):
            DiscretizedGeometry(cylinder, AngularSymmetry(), {"axial": 20})
