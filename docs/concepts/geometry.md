# Geometry

## Why geometry matters

Each unit operation in a simulation has internal states — concentration profiles, volumes,
pressures — that live on some spatial domain. A chromatography column has a bulk concentration
that varies along the axial direction. A stirred tank has a single well-mixed volume. A particle
has a radial concentration gradient.

The shape of this domain, how it is discretized, and which faces connect to the outside world
are all captured by the **geometry** of a state.

Getting geometry right matters because:

- It determines the shape of the state vector (and therefore the solver's `y` vector)
- It defines where inlet and outlet ports are (which faces connect to other unit operations)
- It determines which faces are internal flux interfaces (e.g. bulk ↔ particle)
- It carries the coordinate arrays used for finite-difference residuals and for labeling output

## Two separate concepts

Geometry splits into two orthogonal ideas:

### Physical geometry

The actual shape of the domain — cylinder, sphere, hollow cylinder, frustum. This defines:

- Which dimensions exist (`axial`, `radial`, `angular`, ...)
- The physical extent of each dimension (length, inner/outer radius, ...)
- The boundary type at each face (`inlet`, `outlet`, `wall`, `symmetry`, `flux_interface`)

```python
from CADETPythonSimulator.geometry import Cylinder, Sphere, HollowCylinder, Frustum

col  = Cylinder(length=0.1, radius=0.005)
par  = Sphere(radius=50e-6)
rcol = HollowCylinder(r_inner=0.005, r_outer=0.02)
frus = Frustum(length=0.1, r_inlet=0.01, r_outlet=0.005)
```

### Homogenization

Which dimensions are spatially resolved, and which are collapsed under a symmetry assumption.
A standard 1D column model assumes radial and angular symmetry — only the axial direction is
resolved. A 2D model resolves both axial and radial.

```python
from CADETPythonSimulator.geometry import (
    WellMixed,              # 0D — all dims collapsed
    RadialAngularSymmetry,  # 1D axial
    AngularSymmetry,        # 2D axial + radial
    SphericalSymmetry,      # 1D radial (for particles)
    AxialAngularSymmetry,   # 1D radial (for hollow cylinders)
)
```

:::{important}
Homogenization is not just a grid parameter — it changes the physical model.
A `Column1D` and `Column2D` have different governing equations, not the same equations on
different grids. The homogenization is therefore bundled into the unit operation class and is
not something users change at runtime.
:::

## Discretized geometry

Combining a physical geometry, a homogenization, and a cell count per resolved dimension
produces a `DiscretizedGeometry` — the object that `State` actually uses.

```python
from CADETPythonSimulator.geometry import DiscretizedGeometry

g = DiscretizedGeometry(
    Cylinder(length=0.1, radius=0.005),
    RadialAngularSymmetry(),
    n_cells={"axial": 20},
)

g.shape          # (20,)
g.coordinates    # {"axial": array([0. , ..., 0.1])}
g.external_ports # {"inlet": (0,), "outlet": (-1,)}
g.n_dof          # 20
```

Users never instantiate `DiscretizedGeometry` directly — concrete unit operation classes
(e.g. `Column1D`, `Cstr`) build one internally with sensible defaults.

## Ports

Ports are the faces of the discretized geometry that connect to the outside world.
There are two kinds:

**External ports** — connect to other unit operations at the system level. The `System`
routes flow between them. A 1D column has `inlet` at `s[0]` and `outlet` at `s[-1]`.
A 2D column has `inlet` at `s[0, :]` (a full radial profile) and `outlet` at `s[-1, :]`.

**Flux interfaces** — connect states *within* a unit operation (e.g. bulk ↔ particle in a
GRM column). These are invisible to the `System` and handled inside `compute_residual`.

```python
# 1D column: scalar ports
g1d = DiscretizedGeometry(Cylinder(length=0.1), RadialAngularSymmetry(), {"axial": 20})
g1d.external_ports   # {"inlet": (0,), "outlet": (-1,)}

# 2D column: vector ports (full radial profile)
g2d = DiscretizedGeometry(Cylinder(length=0.1, radius=0.005), AngularSymmetry(), {"axial": 20, "radial": 5})
g2d.external_ports   # {"inlet": (0, slice(None)), "outlet": (-1, slice(None))}

# Sphere: no external ports, only a flux interface at the surface
gs = DiscretizedGeometry(Sphere(radius=50e-6), SphericalSymmetry(), {"radial": 5})
gs.external_ports    # {}
gs.flux_interfaces   # {"flux_interface": (-1,)}
```

## Spatially varying quantities

Some geometries carry physical quantities that vary along a resolved dimension.
The `Frustum` is the main example: because the cross-section `A(z)` changes along the column,
the fluid velocity `u(z) = Q / A(z)` and axial dispersion also vary. The geometry provides
these as numpy arrays so residual functions can use them directly.

```python
g = DiscretizedGeometry(
    Frustum(length=0.1, r_inlet=0.01, r_outlet=0.005),
    RadialAngularSymmetry(),
    {"axial": 10},
)
g.spatially_varying["cross_section"]  # shape (10,), values π r(z)²
```

For all other geometries, `spatially_varying` is empty — residuals can assume constant
cross-section.
