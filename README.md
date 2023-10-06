# Supplementary Information: *Assessment and modeling of the isobaric vapor-liquid-liquid equilibrium for water + cyclopentyl methyl ether + alcohol (ethanol or propan-1-ol) ternary mixtures*

This repository is part of the Supporting Information of the article *Simulation and data-driven modeling of the transport properties of the Mie fluid* by Gustavo Chaparro and Erich A. Müller. (Submitted to Journal of Physical Chemistry B).

This repository includes the following information:
- Databases of the self diffusivity, shear viscosity and thermal conductivity of the Mie fluid computed with Equilibrium Molecular Dynamics.
- Artificial Neural Networks models to compute the transport properties of the Mie Fluid
- Examples of how tho access the database and use the models.


### Databases

The [``databases``](./databases) folder inlcudes the following files:
- `miefluid-diff.csv`: Database of the self diffusivity of the Mie fluid
- `miefluid-tcond.csv`: Database of the shear viscosity of the Mie fluid
- `miefluid-tcond.csv`: Database of the thermal conductivity of the Mie fluid

### Models

The [``models``](./models) folder inlcudes all the trained ANN models. Please refer to the main article for further details aboud each model.

### Examples

The following calculations are included in the [``examples``](./examples) folder.

1. Reading the transport properties databases
2. Loading the ANN models for prediciton of the transport properties of the Mie fluid
3. Example of using the self diffusivity model for the Lennard-Jones fluid
4. Example of using the shear viscosity model for the Lennard-Jones fluid
5. Example of using the thermal conductivity model for the Lennard-Jones fluid

### Prerequisites

- Numpy (tested on version 1.24.2)
- matplotlib (tested on version 3.6.3)
- pandas (tested on version 1.5.3)
- jax (tested on version 0.4.4)
- flax (tested on version 0.6.6)

### License information

See ``LICENSE.md`` for information on the terms & conditions for usage of this software and a DISCLAIMER OF ALL WARRANTIES.

Although not required by the license, if it is convenient for you, please cite this if used in your work. Please also consider contributing any changes you make back, and benefit the community.