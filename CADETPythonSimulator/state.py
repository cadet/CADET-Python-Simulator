from typing import Any, NoReturn
import warnings

import numpy as np

from CADETProcess.dataStructure import Structure, String, SizedNdArray


class State(Structure):
    """
    Manages the state of a system with configurable dimensions and entries.

    Parameters
    ----------
    dimensions : Dict[str, int]
        Mapping of dimension names to their sizes.
    entries : Dict[str, int]
        Mapping of entry names to their counts.
    n_inlet_ports : int | str, optional
        Number of inlet ports or the dimension name for the outlet port, if applicable.
    n_outlet_ports : int | str, optional
        Number of outlet ports or the dimension name for the outlet port, if applicable.

    Attributes
    ----------
    name: str
        The name of the state
    s : np.ndarray
        The state array, initialized as zeros based on the computed shape.
    """

    name = String()
    s = SizedNdArray(size='shape')

    def __init__(
            self,
            name: str,
            dimensions: dict[str, int],
            entries: dict[str, int],
            n_inlet_ports: int = 0,
            n_outlet_ports: int = 0,
            ) -> NoReturn:
        self.name = name
        self.dimensions = dimensions
        self.entries = entries

        if isinstance(n_inlet_ports, str):
            inlet_port_mapping = n_inlet_ports
            n_inlet_ports = self.dimensions[n_inlet_ports]
        else:
            inlet_port_mapping = None
        self.n_inlet_ports = n_inlet_ports
        self.inlet_port_mapping = inlet_port_mapping

        if isinstance(n_outlet_ports, str):
            outlet_port_mapping = n_outlet_ports
            n_outlet_ports = self.dimensions[n_outlet_ports]
        else:
            outlet_port_mapping = None
        self.n_outlet_ports = n_outlet_ports
        self.outlet_port_mapping = outlet_port_mapping

        self.s = np.zeros(self.shape)

    @property
    def n_dimensions(self) -> int:
        """int: Return the number of dimensions."""
        return len(self.dimensions)

    @property
    def dimension_shape(self) -> tuple[int, ...]:
        """tuple of int: Return the shape derived from dimensions."""
        return tuple(self.dimensions.values())

    @property
    def n_cells(self) -> int:
        """int: Return the total number of cells from the product of dimensions."""
        return int(np.prod(self.dimension_shape))

    @property
    def n_entries(self) -> int:
        """int: Return the total number of entries from the sum of entries."""
        return sum(self.entries.values())

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple of int: Return the complete shape of the state array."""
        shape = self.dimension_shape + (self.n_entries,)
        if isinstance(shape, int):
            shape = (shape, )
        return shape

    @property
    def n_dof(self) -> int:
        """int: Return the total number of degrees of freedom."""
        return np.prod(self.shape)

    @property
    def s_flat(self) -> np.ndarray:
        """np.ndarray: Return the state array flattened into one dimension."""
        return self.s.reshape(-1)

    @s_flat.setter
    def s_flat(self, s_flat: np.ndarray) -> NoReturn:
        s = np.array(s_flat).reshape(self.shape)
        self.s = s

    @property
    def s_split(self) -> dict[str, np.ndarray]:
        """
        Dict where each key is an entry name and the value is the corresponding value.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary where each key is an entry name and the value is the
            corresponding segment of the state array.
        """
        s_split = {}
        start_index = 0
        for entry, n_entries in self.entries.items():
            end_index = start_index + n_entries
            s_split[entry] = self.s[..., start_index:end_index]
            start_index = end_index

        return s_split

    @s_split.setter
    def s_split(self, s_split: dict[str, np.ndarray]):
        _s_split = self.s_split

        for key, value in s_split.items():
            _s_split[key][:] = value[:]

    def validate_coupling_structure(self, coupling_structure: dict[str, int]) -> bool:
        """
        Validate coupling structure is correct.

        This method validates that all entries of the coupling structure are present
        in the state entries.

        Parameters
        ----------
        coupling_structure : dict[str, int]
            Dict containing the keys and number of entries of the coupling structure.

        Returns
        -------
        bool
            True if coupling structure is valid, False otherwise.

        """
        flag = True
        for entry, n_entries in coupling_structure:
            if entry not in self.entries:
                flag = False
                warnings.warn(f"Entry '{entry}' is not found in entries.")
            if n_entries != self.entries[entry]:
                flag = False
                warnings.warn(
                    f"Entry '{entry}' does not have shape {self.entries[entry]}."
                )
        return flag

    def set_inlet_port_state(
            self,
            inlet_port_state: dict[str, np.ndarray],
            port_index: int
            ) -> NoReturn:
        """
        Set the state for a specified inlet port.

        Parameters
        ----------
        inlet_port_state : Dict[str, np.ndarray]
            A dictionary mapping each state entry to its new values at the inlet port.
        port_index : int
            The index of the port in the inlet port dimension to update.

        Raises
        ------
        Exception
            If port index exceeds number of inlet ports.
        """
        if port_index > (self.n_inlet_ports - 1):
            raise Exception("Port index exceeds number of inlet ports.")

        # Create a slicing object dynamically
        slice_indices = [0] * (self.n_dimensions + 1)

        port_dim_index = self.inlet_port_mapping or 0
        if isinstance(port_dim_index, str):
            dim_keys = list(self.dimensions.keys())
            port_dim_index = dim_keys.index(port_dim_index)

        # Set the slice for the port dimension to the specific index
        slice_indices[port_dim_index] = port_index

        # State entries are always the last dimension, so set it to slice all entries
        slice_indices[-1] = slice(None)

        # Convert list of slices to a tuple (as numpy expects a tuple for slicing)
        slice_tuple = tuple(slice_indices)

        # Assemble
        s_split = self.s_split
        for component, entry in inlet_port_state.items():
            s_split[component][slice_tuple] = entry

    def get_outlet_port_state(
            self,
            port_index: int
            ) -> dict[str, np.ndarray]:
        """
        Retrieve the state for a specified outlet port.

        Parameters
        ----------
        port_index : int
            The index of the port in the outlet port dimension to retrieve.

        Returns
        -------
        Dict[str, np.ndarray]
            A dictionary mapping each state entry to its current values at the outlet
            port.

        Raises
        ------
        Exception
            If port index exceeds number of outlet ports.
        """
        if port_index > (self.n_outlet_ports - 1):
            raise ValueError("Port index exceeds number of outlet ports.")

        # Create a slicing object dynamically
        slice_indices = [-1] * (self.n_dimensions + 1)

        port_dim_index = self.outlet_port_mapping or -1
        if isinstance(port_dim_index, str):
            dim_keys = list(self.dimensions.keys())
            port_dim_index = dim_keys.index(port_dim_index)

        # Set the slice for the port dimension to the specific index
        slice_indices[port_dim_index] = port_index

        # State entries are always the last dimension, so set it to slice all entries
        slice_indices[-1] = slice(None)

        # Convert list of slices to a tuple (as numpy expects a tuple for slicing)
        slice_tuple = tuple(slice_indices)

        # Assemble
        s_split = self.s_split
        outlet_port_state = {}

        for component, n_entries in self.entries.items():
            outlet_port_state[component] = s_split[component][slice_tuple]

        return outlet_port_state

    def __str__(self) -> str:
        """str: String representation of the State instance."""
        return self.name

    def __repr__(self):
        n_inlet_ports = self.n_inlet_ports
        port_mapping = self.inlet_port_mapping
        if n_inlet_ports > 0 and port_mapping is not None:
            n_inlet_ports = port_mapping
            n_inlet_ports = f"'{port_mapping}'"

        n_outlet_ports = self.n_outlet_ports
        port_mapping = self.outlet_port_mapping
        if n_outlet_ports > 0 and port_mapping is not None:
            n_outlet_ports = f"'{port_mapping}'"

        return (
            f"State(name='{self.name}', "
            f"dimensions={self.dimensions}, "
            f"entries={self.entries}, "
            f"n_inlet_ports={n_inlet_ports}, "
            f"n_outlet_ports={n_outlet_ports})"
        )


# %%

def state_factory(
        instance: Any,
        name: str,
        dimensions: tuple[int],
        entries: dict[str, int | str],
        n_inlet_ports: int | str = 0,
        n_outlet_ports: int | str = 0,
        ) -> State:
    """
    Initialize a State instance.

    Parameters
    ----------
    name : str
        Name of the state
    dimensions : int or tuple
        The expected dimensions of the state. Individual elements can be either
        integers or strings (indicating other instance parameters).
    entries : dict
        The state entries at each cell. Individual elements can be either
        integers or strings (indicating other instance parameters).
    """
    dimensions = {
        dim: getattr(instance, dim) for dim in dimensions
    }
    entries = {
        entry: (
            n_entries if isinstance(n_entries, int) else getattr(instance, n_entries)
        )
        for entry, n_entries in entries.items()
    }

    state = State(
        name=name,
        dimensions=dimensions,
        entries=entries,
        n_inlet_ports=n_inlet_ports,
        n_outlet_ports=n_outlet_ports,
    )

    return state
