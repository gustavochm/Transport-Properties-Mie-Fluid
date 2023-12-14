from typing import Any, Callable, Sequence
from jax import numpy as jnp
from flax import linen as nn
from .flax_utils import linear_activation
from .jax_utils import val_and_jacrev, val_and_jacfwd
from jax import vmap
from .HelmholtzModel import helper_get_alpha
Dtype = Any


class TransportModelResidual_PVT_Tinv(nn.Module):
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

    def __call__(self, lambda_r, rhoad, Tad):

        alpha = helper_get_alpha(lambda_r, lambda_a=6.)

        x = jnp.stack([alpha, rhoad, 1./Tad]).T

        rhoad0 = jnp.zeros_like(rhoad)
        x0 = jnp.stack([alpha, rhoad0, 1./Tad]).T

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = self.hidden_activation(x)

            x0 = layer(x0)
            x0 = self.hidden_activation(x0)

        x = self.transport_layer(x)
        x = self.output_activation(x)

        x0 = self.transport_layer(x0)
        x0 = self.output_activation(x0)

        x = x - x0
        x = x.flatten()

        return x

    def dtransport_drho(self, lambda_r, rhoad, Tad):
        fun = vmap(val_and_jacrev(self.__call__, argnums=1, has_aux=False),
                   in_axes=(0, 0, 0), out_axes=0)
        dtransport_drho, transport = fun(lambda_r, rhoad, Tad)
        dtransport_drho = dtransport_drho.reshape(rhoad.shape)
        transport = transport.reshape(rhoad.shape)
        return transport, dtransport_drho

    def d2transport_drho2(self, lambda_r, rhoad, Tad):
        fun = vmap(val_and_jacfwd(val_and_jacrev(self.__call__, argnums=1,
                   has_aux=False), argnums=1, has_aux=True),
                   in_axes=(0, 0, 0), out_axes=0)
        out = fun(lambda_r, rhoad, Tad)
        d2transport_drho2, (dtransport_drho, transport) = out
        d2transport_drho2 = d2transport_drho2.reshape(rhoad.shape)
        dtransport_drho = dtransport_drho.reshape(rhoad.shape)
        transport = transport.reshape(rhoad.shape)
        return transport, dtransport_drho, d2transport_drho2
