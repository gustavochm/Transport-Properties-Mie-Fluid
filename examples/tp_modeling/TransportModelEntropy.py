from typing import Any, Callable, Sequence
import jax
import optax
from jax import numpy as jnp
from flax import linen as nn
import numpy as np
from .flax_utils import linear_activation

Dtype = Any


class TransportModel_entropy(nn.Module):
    features: Sequence[int]
    dtype: Dtype = jnp.float64
    hidden_activation: Callable = nn.tanh
    output_activation: Callable = linear_activation

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.number_layers = len(self.features)

        self.hidden_layers = [nn.Dense(feat, use_bias=True, dtype=self.dtype,                       
                                       kernel_init=nn.initializers.glorot_uniform(dtype=self.dtype),
                                       param_dtype=self.dtype) for feat in self.features]
        self.transport_layer = nn.Dense(1, use_bias=True, dtype=self.dtype, 
                                        kernel_init=nn.initializers.glorot_uniform(dtype=self.dtype),
                                        param_dtype=self.dtype)

    def __call__(self, alpha, entropy):

        x = jnp.stack([alpha, entropy]).T

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.hidden_activation(x)
        x = self.transport_layer(x)
        x = self.output_activation(x)
        x = x.flatten()

        return x


def diffusivity_scaling(rhoad: Sequence, Tad: Sequence, diff: Sequence,
                        unscale: bool = False) -> Sequence:
    """Scaling function for diffusivity

    Args:
        rhoad (Sequence): Reduced density
        Tad (Sequence): Reduced temperature
        diff (Sequence): Diffusivity
        unscale (bool, optional): If True, unscale the diffusivity. Defaults to False.

    Returns:
        Sequence: Scaled diffusivity
    """
    if unscale:
        # revert the scaling
        diffusivity = diff * (rhoad**(2/3) * np.sqrt(Tad)) / rhoad
    else:
        # scale the diffusivity
        diffusivity = rhoad*diff / (rhoad**(2/3) * np.sqrt(Tad))

    return diffusivity


def viscosity_scaling(rhoad: Sequence, Tad: Sequence, visc: Sequence,
                      unscale: bool = False) -> Sequence:
    """Scaling function for viscosity

    Args:
        rhoad (Sequence): Reduced density
        Tad (Sequence): Reduced temperature
        visc (Sequence): viscosity
        unscale (bool, optional): If True, unscale the viscosity. Defaults to False.

    Returns:
        Sequence: Scaled diffusivity
    """

    if unscale:
        # revert the scaling
        viscosity = visc * (rhoad**(2/3) * np.sqrt(Tad))
    else:
        # scale the viscosity
        viscosity = visc / (rhoad**(2/3) * np.sqrt(Tad))

    return viscosity


def thermal_conductivity_scaling(rhoad: Sequence, Tad: Sequence, tcond: Sequence,
                                 unscale: bool = False) -> Sequence:
    """Scaling function for viscosity

    Args:
        rhoad (Sequence): Reduced density
        Tad (Sequence): Reduced temperature
        tcond (Sequence): thermal conductivity
        unscale (bool, optional): If True, unscale the viscosity. Defaults to False.

    Returns:
        Sequence: Scaled diffusivity
    """

    if unscale:
        # revert the scaling
        thermal_conductivity = tcond * (rhoad**(2/3) * np.sqrt(Tad))
    else:
        # scale the thermal conductivity
        thermal_conductivity = tcond / (rhoad**(2/3) * np.sqrt(Tad))

    return thermal_conductivity
