from CADETProcess.processModel import ComponentSystem, Component, Species
from CADETProcess.dataStructure import UnsignedFloat, String, Integer
from CADETProcess.dataStructure import Structure

from CADETPythonSimulator.exception import CADETPythonSimError
from functools import wraps


class CPSSpecies(Structure):
    """
    Species class.

    Represent a species in a chemical system.
    Same as in cadet Process but with added
    density and Volume

    Attributes
    ----------
    name : str
        The name of the species.
    charge : int, optional
        The charge of the species. Default is 0.
    molecular_weight : float
        The molecular weight of the species.
    density : float
        Density of the species.
    molecular_volume : float
        The molecular volume of the species
    specific_cake_resistance : float
        The specific cake resistance of the species

    """

    name = String()
    charge = Integer(default=0)
    molecular_weight = UnsignedFloat()
    density = UnsignedFloat()
    molecular_volume = UnsignedFloat()
    viscosity = UnsignedFloat()
    pure_density = UnsignedFloat()
    specific_cake_resistance = UnsignedFloat()


class CPSComponent(Component):
    """
    Information about single component.

    Inherits from CadetProcess Component
    Same function but with fixed molecular weight and added densities and volume.
    A component can contain subspecies (e.g. differently charged variants).

    Attributes
    ----------
    name : String
        Name of the component.
    species : list
        List of Subspecies.
    n_species : int
        Number of Subspecies.
    label : list
        Name of component (including species).
    charge : list
        Charge of component (including species).
    molecular_weight : list
        Molecular weight of component (including species).
    density : list
        density of component (including species).
    molecular_volume : list
        Molecular volume of component (including species).
    viscosity : list
        Viscosity of component(including species)
    pure_density : list
        pure_density of component(including species)
    specific_cake_resistance : list
        specific_cake_resistance of component(including species)

    See Also
    --------
    Species
    ComponentSystem

    """

    def __init__(self,
                 name=None,
                 species=None,
                 charge=None,
                 molecular_weight=None,
                 density=None,
                 molecular_volume=None,
                 viscosity=None,
                 pure_density=None,
                 specific_cake_resistance=None):
        """Construct CPSComponent."""
        self.name = name
        self._species = []

        if species is None:
            self.add_species(
                name,
                charge,
                molecular_weight,
                density,
                molecular_volume,
                viscosity,
                pure_density,
                specific_cake_resistance
            )
        elif isinstance(species, str):
            self.add_species(
                species,
                charge,
                molecular_weight,
                density,
                molecular_volume,
                viscosity,
                pure_density,
                specific_cake_resistance
            )
        elif isinstance(species, list):
            if charge is None:
                charge = len(species) * [None]
            if molecular_weight is None:
                molecular_weight = len(species) * [None]
            if density is None:
                density = len(species) * [None]
            if molecular_volume is None:
                molecular_volume = len(species) * [None]
            if viscosity is None:
                viscosity = len(species) * [None]
            if pure_density is None:
                pure_density = len(species) * [None]
            if specific_cake_resistance is None:
                specific_cake_resistance = len(species) * [None]
            for i, spec in enumerate(species):
                self.add_species(
                    spec,
                    charge[i],
                    molecular_weight[i],
                    density[i],
                    molecular_volume[i],
                    viscosity[i],
                    pure_density[i],
                    specific_cake_resistance[i]
                )
        else:
            raise CADETPythonSimError("Could not determine number of species")

    def add_species(self, species, *args, **kwargs):
        """Add a species to the component System."""
        if not isinstance(species, CPSSpecies):
            species = CPSSpecies(species, *args, **kwargs)
        self._species.append(species)

    @property
    def molecular_volume(self):
        """List of float or None: The molecular volume of the subspecies."""
        return [spec.molecular_volume for spec in self.species]

    @property
    def density(self):
        """List of float or None: The density of the subspecies."""
        return [spec.density for spec in self.species]

    @property
    def molecular_weight(self):
        """List of float or None: The molecular weights of the subspecies."""
        return [spec.molecular_weight for spec in self.species]

    @property
    def viscosity(self):
        """List of float or None: The viscosity of the subspecies."""
        return [spec.viscosity for spec in self.species]

    @property
    def pure_density(self):
        """List of float or None: Density of the subspecies."""
        return [spec.pure_density for spec in self.species]

    @property
    def specific_cake_resistance(self):
        """List of float or None: specific cake resistance of the subspecies."""
        return [spec.specific_cake_resistance for spec in self.species]

class CPSComponentSystem(ComponentSystem):
    """
    Component System Class.

    Information about components in system. Inherits from Component System. Adds
    molecular Volume to the Component System.

    A component can contain subspecies (e.g. differently charged variants).

    Attributes
    ----------
    name : String
        Name of the component system.
    components : list
        List of individual components.
    n_species : int
        Number of Subspecies.
    n_comp : int
        Number of all component species.
    n_components : int
        Number of components.
    indices : dict
        Component indices.
    names : list
        Names of all components.
    species : list
        Names of all component species.
    charge : list
        Charges of all components species.
    molecular_weight : list
        Molecular weights of all component species.
    molecular_volume : list
        Molecular volume of all component species.
    viscosity : list
        Viscosity of all component species.
    pure_density : list
        Pure density of all component species.

    See Also
    --------
    Species
    Component

    """

    def __init__(
            self,
            components=None,
            name=None,
            charges=None,
            molecular_weights=None,
            densities=None,
            molecular_volume=None,
            viscosities=None,
            pure_densities=None,
            specific_cake_resistances=None
        ):
        """
        Initialize the ComponentSystem object.

        Parameters
        ----------
        components : int, list, None
            The number of components or the list of components to be added.
            If None, no components are added.
        name : str, None
            The name of the ComponentSystem.
        charges : list, None
            The charges of each component.
        molecular_weights : list, None
            The molecular weights of each component.
        densities : list, None
            Densities of each component
        molecular_volume : list, None
            The molecular volume of each component.
        viscosities : list, None
            viscosity of each component.
        pure_densities : list, None
            pure densities of each component.
        specific_cake_resistances : list, None.
            specific cake resistance of each component

        Raises
        ------
        CADETProcessError
            If the `components` argument is neither an int nor a list.

        """
        self.name = name

        self._components = []

        if components is None:
            return

        if isinstance(components, int):
            n_comp = components
            components = [str(i) for i in range(n_comp)]
        elif isinstance(components, list):
            n_comp = len(components)
        else:
            raise CADETPythonSimError("Could not determine number of components")

        if charges is None:
            charges = n_comp * [None]
        if molecular_weights is None:
            molecular_weights = n_comp * [None]
        if densities is None:
            densities = n_comp * [None]
        if molecular_volume is None:
            molecular_volume = n_comp * [None]
        if viscosities is None:
            viscosities = n_comp * [None]
        if pure_densities is None:
            pure_densities = n_comp * [None]
        if specific_cake_resistances is None:
            specific_cake_resistances = n_comp * [None]


        for i, comp in enumerate(components):
            self.add_component(
                comp,
                charge=charges[i],
                molecular_weight=molecular_weights[i],
                density=densities[i],
                molecular_volume=molecular_volume[i],
                viscosity=viscosities[i],
                pure_density=pure_densities[i],
                specific_cake_resistance=specific_cake_resistances[i]
            )

    @wraps(CPSComponent.__init__)
    def add_component(self, component, *args, **kwargs):
        """
        Add a component to the system.

        Parameters
        ----------
        component : {str, Component}
            The class of the component to be added.
        *args : list
            The positional arguments to be passed to the component class's constructor.
        **kwargs : dict
            The keyword arguments to be passed to the component class's constructor.

        """
        if not isinstance(component, CPSComponent):
            component = CPSComponent(component, *args, **kwargs)

        if component.name in self.names:
            raise CADETPythonSimError(
                f"Component '{component.name}' "
                "already exists in ComponentSystem."
            )

        self._components.append(component)

    @property
    def molecular_volumes(self):
        """list: List of species molecular volumes."""
        molecular_volumes = []
        for comp in self.components:
            molecular_volumes += comp.molecular_volume

        return molecular_volumes

    @property
    def densities(self):
        """list: List of species densities."""
        densities = []
        for comp in self.components:
            densities += comp.density

        return densities

    @property
    def viscosities(self):
        """list: List of species viscosity."""
        viscosities = []
        for comp in self.components:
            viscosities +=comp.viscosity

        return viscosities

    @property
    def pure_densities(self):
        """list: List of species density if pure."""
        pure_densities = []
        for comp in self.components:
            pure_densities += comp.pure_density

        return pure_densities

    @property
    def specific_cake_resistances(self):
        """list: List of species cake resistance if pure."""
        specific_cake_resistances = []
        for comp in self.components:
            specific_cake_resistances += comp.specific_cake_resistance

        return specific_cake_resistances
