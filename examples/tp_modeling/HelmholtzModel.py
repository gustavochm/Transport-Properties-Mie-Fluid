from typing import Any, Sequence
import jax
from jax import vmap
from jax import numpy as jnp
from .jax_utils import val_and_jacrev, val_and_jacfwd
from flax import linen as nn

Dtype = Any

class HelmholtzModel(nn.Module):
    """
    Helmholtz free energy model based on artificial neural networks for the Mie fluid

    Parameters
    ----------
    features : Sequence[int]
        Number of neurons in each hidden layer.
    dtype : Dtype, optional
        Data type for the model. The default is jnp.float64.
    """
    features: Sequence[int]
    dtype: Dtype = jnp.float64

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.number_layers = len(self.features)

        self.hidden_layers = [nn.Dense(feat, use_bias=True, dtype=self.dtype,                          
                                       kernel_init=nn.initializers.glorot_uniform(dtype=self.dtype), 
                                       param_dtype=self.dtype) for feat in self.features]
        self.helmholtz_layer = nn.Dense(1, use_bias=False, dtype=self.dtype, 
                                        kernel_init=nn.initializers.glorot_uniform(dtype=self.dtype), 
                                        param_dtype=self.dtype)

    def __call__(self, alpha, rhoad, Tad):
        rhoad0 = jnp.zeros_like(rhoad)

        x = jnp.stack([alpha, rhoad, 1./Tad]).T
        x_rhoad0 = jnp.stack([alpha, rhoad0, 1./Tad]).T

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = nn.tanh(x)

            x_rhoad0 = layer(x_rhoad0)
            x_rhoad0 = nn.tanh(x_rhoad0)

        x = self.helmholtz_layer(x)
        x_rhoad0 = self.helmholtz_layer(x_rhoad0)
        helmholtz = (x - x_rhoad0).flatten()

        return helmholtz

    def dhelmholtz_drho(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacrev(self.__call__, argnums=1, has_aux=False),
                   in_axes=(0, 0, 0), out_axes=0)
        dAres_drho, Ares = fun(alpha, rhoad, Tad)
        dAres_drho = dAres_drho.reshape(rhoad.shape)
        Ares = Ares.reshape(rhoad.shape)
        return Ares, dAres_drho

    def dhelmholtz_dT(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacrev(self.__call__, argnums=2, has_aux=False),
                   in_axes=(0, 0, 0), out_axes=0) 
        dAres_dT, Ares = fun(alpha, rhoad, Tad)
        dAres_dT = dAres_dT.reshape(Tad.shape)
        Ares = Ares.reshape(Tad.shape)
        return Ares, dAres_dT

    def dhelmholtz(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacrev(self.__call__, argnums=[1, 2], has_aux=False),
                   in_axes=(0, 0, 0), out_axes=0) 
        (dAres_drho, dAres_dT), Ares = fun(alpha, rhoad, Tad)
        dAres_drho = dAres_drho.reshape(rhoad.shape)
        dAres_dT = dAres_dT.reshape(Tad.shape)
        Ares = Ares.reshape(Tad.shape)
        return Ares, dAres_drho, dAres_dT

    def d2helmholtz_drho2(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacfwd(val_and_jacrev(self.__call__, argnums=1,
                  has_aux=False), argnums=1, has_aux=True), in_axes=(0, 0, 0), out_axes=0)
        d2Ares_drho2, (dAres_drho, Ares) = fun(alpha, rhoad, Tad)
        d2Ares_drho2 = d2Ares_drho2.reshape(rhoad.shape)
        dAres_drho = dAres_drho.reshape(rhoad.shape)
        Ares = Ares.reshape(rhoad.shape)
        return Ares, dAres_drho, d2Ares_drho2

    def d2helmholtz_drho_dT(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacfwd(val_and_jacrev(self.__call__, argnums=1,
                  has_aux=False), argnums=2, has_aux=True), in_axes=(0, 0, 0), out_axes=0)
        d2Ares_drho_dT, (dAres_drho, Ares) = fun(alpha, rhoad, Tad)
        d2Ares_drho_dT = d2Ares_drho_dT.reshape(rhoad.shape)
        dAres_drho = dAres_drho.reshape(rhoad.shape)
        Ares = Ares.reshape(rhoad.shape)
        return Ares, dAres_drho, d2Ares_drho_dT

    def d2helmholtz_drho2_dT(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacfwd(val_and_jacrev(self.__call__, argnums=1,
                  has_aux=False), argnums=[1, 2], has_aux=True), in_axes=(0, 0, 0), out_axes=0)
        (d2Ares_drho2, d2Ares_drho_dT), (dAres_drho, Ares) = fun(alpha, rhoad, Tad)
        d2Ares_drho2 = d2Ares_drho2.reshape(rhoad.shape)
        d2Ares_drho_dT = d2Ares_drho_dT.reshape(rhoad.shape)
        dAres_drho = dAres_drho.reshape(rhoad.shape)
        Ares = Ares.reshape(rhoad.shape)
        return Ares, dAres_drho, d2Ares_drho2, d2Ares_drho_dT

    def d2helmholtz_dT2(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacfwd(val_and_jacrev(self.__call__, argnums=2,
                  has_aux=False), argnums=2, has_aux=True), in_axes=(0, 0, 0), out_axes=0)
    
        d2Ares_dT2, (dAres_dT, Ares) = fun(alpha, rhoad, Tad)
        d2Ares_dT2 = d2Ares_dT2.reshape(Tad.shape)
        dAres_dT = dAres_dT.reshape(Tad.shape)
        Ares = Ares.reshape(Tad.shape)
        return Ares, dAres_dT, d2Ares_dT2

    def d2helmholtz(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacfwd(val_and_jacrev(self.__call__, argnums=[1, 2],
                  has_aux=False), argnums=[1, 2], has_aux=True), in_axes=(0, 0, 0), out_axes=0)

        out = fun(alpha, rhoad, Tad)
        ((d2Ares_drho2, d2Ares_drho_dT), (_, d2Ares_dT2)) = out[0]
        (dAres_drho, dAres_dT), Ares = out[1]

        d2Ares_drho2 = d2Ares_drho2.reshape(rhoad.shape)
        d2Ares_dT2 = d2Ares_dT2.reshape(Tad.shape)
        d2Ares_drho_dT = d2Ares_drho_dT.reshape(Tad.shape)
        dAres_drho = dAres_drho.reshape(rhoad.shape)
        dAres_dT = dAres_dT.reshape(Tad.shape)
        Ares = Ares.reshape(Tad.shape)

        return Ares, dAres_drho, dAres_dT, d2Ares_drho2, d2Ares_dT2, d2Ares_drho_dT

    def d3helmholtz_drho3(self, alpha, rhoad, Tad):
        fun = vmap(val_and_jacrev( val_and_jacrev( val_and_jacrev(
                   self.__call__, argnums=1, has_aux=False),
                   argnums=1, has_aux=True),
                   argnums=1, has_aux=True), in_axes=(0, 0, 0))  
        d3Ares_drho3, (d2Ares_drho2, (dAres_drho, Ares)) = fun(alpha, rhoad, Tad)
        d3Ares_drho3 = d3Ares_drho3.reshape(rhoad.shape)
        d2Ares_drho2 = d2Ares_drho2.reshape(rhoad.shape)
        dAres_drho = dAres_drho.reshape(rhoad.shape)
        Ares = Ares.reshape(rhoad.shape)
        return Ares, dAres_drho, d2Ares_drho2, d3Ares_drho3

    ######################################
    # Total: Residual + Ideal properties #
    ######################################

    def pressure(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho = self.apply(params, alpha, rhoad, Tad, method=self.dhelmholtz_drho)
        # Ares, dAres_drho = self.dhelmholtz_drho(params, alpha, rhoad, Tad)

        Pad_res = rhoad**2 * dAres_drho
        Pad_id = rhoad * Tad  # ideal gas
        # Total Pressure
        Pad = Pad_res + Pad_id
        return Pad

    def dpressure_drho(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho, d2Ares_drho2 = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz_drho2)
        # Ares, dAres_drho, d2Ares_drho2 = self.d2helmholtz_drho2(params, alpha, rhoad, Tad)

        # Pressure calculation
        Pad_res = rhoad**2 * dAres_drho
        Pad_id = rhoad * Tad  # ideal gas
        # Total Pressure
        Pad = Pad_res + Pad_id

        # dPressure_drho calculation
        dP_drho_res = 2. * rhoad * dAres_drho + rhoad**2 * d2Ares_drho2
        dP_drho_id = 1. * Tad
        dP_drho = dP_drho_id + dP_drho_res
        return Pad, dP_drho

    def dpressure_dT(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho, d2Ares_drho_dT = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz_drho_dT)
        # Ares, dAres_drho, d2Ares_drho_dT = self.d2helmholtz_drho_dT(params, alpha, rhoad, Tad)

        # Pressure calculation
        Pad_res = rhoad**2 * dAres_drho
        Pad_id = rhoad * Tad  # ideal gas
        # Total Pressure
        Pad = Pad_res + Pad_id

        # Pressure derivatives
        dP_dT_by_rhoad_res = rhoad * d2Ares_drho_dT
        dP_dT_by_rhoad_id = 1. 
        dP_dT_by_rhoad = dP_dT_by_rhoad_id +  dP_dT_by_rhoad_res
        dP_dT = dP_dT_by_rhoad * rhoad 
        return Pad, dP_dT

    def dpressure(self, params, alpha, rhoad, Tad):
        out = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz_drho2_dT)
        # out = self.d2helmholtz_drho2_dT(params, alpha, rhoad, Tad)
        Ares, dAres_drho, d2Ares_drho2, d2Ares_drho_dT = out

        # Pressure calculation
        Pad_res = rhoad**2 * dAres_drho
        Pad_id = rhoad * Tad  # ideal gas 
        # Total Pressure
        Pad = Pad_res + Pad_id

        # Pressure derivatives
        dP_drho_res = 2. * rhoad * dAres_drho + rhoad**2 * d2Ares_drho2
        dP_drho_id = 1. * Tad
        dP_drho = dP_drho_id + dP_drho_res

        dP_dT_by_rhoad_res = rhoad * d2Ares_drho_dT
        dP_dT_by_rhoad_id = 1. 
        dP_dT_by_rhoad = dP_dT_by_rhoad_id +  dP_dT_by_rhoad_res
        dP_dT = dP_dT_by_rhoad * rhoad 
        return Pad, dP_drho, dP_dT

    def d2pressure_drho2(self, params, alpha, rhoad, Tad):
        out = self.apply(params, alpha, rhoad, Tad, method=self.d3helmholtz_drho3)
        # out = self.d3helmholtz_drho3(params, alpha, rhoad, Tad)
        Ares, dAres_drho, d2Ares_drho2, d3Ares_drho3 = out

        # Pressure calculation
        Pad_res = rhoad**2 * dAres_drho
        Pad_id = rhoad * Tad  # ideal gas
        # Total Pressure
        Pad = Pad_res + Pad_id

        # dPressure_drho calculation
        dP_drho_res = 2. * rhoad * dAres_drho + rhoad**2 * d2Ares_drho2
        dP_drho_id = 1. * Tad
        dP_drho = dP_drho_id + dP_drho_res

        # d2Pressure_drho2 calculation
        d2P_drho2_res1 = 2. * (dAres_drho + rhoad*d2Ares_drho2)
        d2P_drho2_res2 = 2 * rhoad * d2Ares_drho2 + rhoad**2 * d3Ares_drho3
        d2P_drho2_res = d2P_drho2_res1 + d2P_drho2_res2
        d2P_drho2_id = 0.
        d2P_drho2 = d2P_drho2_id + d2P_drho2_res
        return Pad, dP_drho, d2P_drho2

    def chemical_potential(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho = self.apply(params, alpha, rhoad, Tad, method=self.dhelmholtz_drho)
        # Ares, dAres_drho = self.dhelmholtz_drho(params, alpha, rhoad, Tad)

        # Residual Chemical Potential
        chem_pot_res = Ares + rhoad * dAres_drho

        # Ideal Chemical Potential
        Aid = Tad*(jnp.log(rhoad) - 1.)
        # dAid_drho = Tad/rhoad
        # chem_pot_id = Aid + rhoad*dAid_drho
        chem_pot_id = Aid + Tad

        chem_pot = chem_pot_id + chem_pot_res
        return chem_pot

    def pressure_and_chempot(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho = self.apply(params, alpha, rhoad, Tad, method=self.dhelmholtz_drho)
        # Ares, dAres_drho = self.dhelmholtz_drho(params, alpha, rhoad, Tad)

        # Pressure calculation
        Pad_res = rhoad**2 * dAres_drho
        Pad_id = rhoad * Tad  # ideal gas
        # Total Pressure
        Pad = Pad_res + Pad_id

        # Residual Chemical Potential
        chem_pot_res = Ares + rhoad * dAres_drho

        # Ideal Chemical Potential
        Aid = Tad*(jnp.log(rhoad) - 1.)
        # dAid_drho = Tad/rhoad
        # chem_pot_id = Aid + rhoad*dAid_drho
        chem_pot_id = Aid + Tad

        chem_pot = chem_pot_id + chem_pot_res
        return Pad, chem_pot

    #######################
    # Residual properties #
    #######################

    def chemical_potential_residual(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho = self.apply(params, alpha, rhoad, Tad, method=self.dhelmholtz_drho)
        # Residual Chemical Potential
        chem_pot_res = Ares + rhoad * dAres_drho
        return chem_pot_res

    def entropy_residual(self, params, alpha, rhoad, Tad):
        Ares, dAres_dT = self.apply(params, alpha, rhoad, Tad, method=self.dhelmholtz_dT)
        # Ares, dAres_dT = self.dhelmholtz_dT(params, alpha, rhoad, Tad)

        # Residual Entropy
        entropy_res = - dAres_dT
        return entropy_res

    def internal_energy_residual(self, params, alpha, rhoad, Tad):
        Ares, dAres_dT = self.apply(params, alpha, rhoad, Tad, method=self.dhelmholtz_dT)
        # Ares, dAres_dT = self.dhelmholtz_dT(params, alpha, rhoad, Tad)

        # Residual Internal Energy
        internal_res = Ares - Tad * dAres_dT
        return internal_res

    def enthalpy_residual(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho, dAres_dT = self.apply(params, alpha, rhoad, Tad, method=self.dhelmholtz)
        # Ares, dAres_drho, dAres_dT = self.dhelmholtz(params, alpha, rhoad, Tad)

        internal_res = Ares - Tad * dAres_dT
        PV_res = rhoad * dAres_drho
        # Residual Enthalpy
        enthalpy_res = internal_res + PV_res
        return enthalpy_res

    def gibbs_residual(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho, dAres_dT = self.apply(params, alpha, rhoad, Tad, method=self.dhelmholtz)
        # Ares, dAres_drho, dAres_dT = self.dhelmholtz(params, alpha, rhoad, Tad)

        internal_res = Ares - Tad * dAres_dT
        PV_res = rhoad * dAres_drho
        # Residual Enthalpy
        enthalpy_res = internal_res + PV_res

        #residual Entropy
        entropy_res = - dAres_dT

        gibbs_res = enthalpy_res - Tad * entropy_res
        return gibbs_res

    def cv_residual(self, params, alpha, rhoad, Tad):
        Ares, dAres_dT, d2Ares_dT2 = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz_dT2)
        # Ares, dAres_dT, d2Ares_dT2 = self.d2helmholtz_dT2(params, alpha, rhoad, Tad)

        # Residual Isochoric heat capacity
        cv_res = - Tad * d2Ares_dT2
        return cv_res

    def cp_residual(self, params, alpha, rhoad, Tad):
        out = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz)
        # out = self.d2helmholtz(params, alpha, rhoad, Tad)
        Ares, dAres_drho, dAres_dT, d2Ares_drho2, d2Ares_dT2, d2Ares_drho_dT = out

        # Isochoric heat capacity
        cv_res = - Tad * d2Ares_dT2

        # Pressure derivatives
        dP_dT_by_rhoad_res = rhoad * d2Ares_drho_dT
        dP_dT_by_rhoad_id = 1. 
        dP_dT_by_rhoad = dP_dT_by_rhoad_id +  dP_dT_by_rhoad_res 

        dP_drho_res = 2. * rhoad * dAres_drho + rhoad**2 * d2Ares_drho2
        dP_drho_id = 1. * Tad
        dP_drho = dP_drho_id + dP_drho_res

        # Thermal Expansion Coefficient
        alphap = dP_dT_by_rhoad / dP_drho

        # Isothermal Compressibility
        rho_kappaT = 1. / dP_drho

        # Residual Isobaric heat capacity
        cp_res = cv_res + Tad * alphap**2 / rho_kappaT - 1.
        return cp_res

    ###################################
    # Other Thermophysical properties #
    ###################################

    def thermal_expansion_coeff(self, params, alpha, rhoad, Tad):
        out = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz_drho2_dT)
        # out = self.d2helmholtz_drho2_dT(params, alpha, rhoad, Tad)
        Ares, dAres_drho, d2Ares_drho2, d2Ares_drho_dT = out

        # Pressure derivatives
        dP_dT_by_rhoad_res = rhoad * d2Ares_drho_dT
        dP_dT_by_rhoad_id = 1. 
        dP_dT_by_rhoad = dP_dT_by_rhoad_id +  dP_dT_by_rhoad_res 

        dP_drho_res = 2. * rhoad * dAres_drho + rhoad**2 * d2Ares_drho2
        dP_drho_id = 1. * Tad
        dP_drho = dP_drho_id + dP_drho_res

        # Thermal Expansion Coefficient
        alphap = dP_dT_by_rhoad / dP_drho

        return alphap

    def isothermal_compressibility(self, params, alpha, rhoad, Tad):
        Ares, dAres_drho, d2Ares_drho2 = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz_drho2)
        # Ares, dAres_drho, d2Ares_drho2 = self.d2helmholtz_drho2(params, alpha, rhoad, Tad)

        # Pressure derivatives
        dP_drho_res = 2. * rhoad * dAres_drho + rhoad**2 * d2Ares_drho2
        dP_drho_id = 1. * Tad
        dP_drho = dP_drho_id + dP_drho_res

        # Isothermal Compressibility
        rho_kappaT = 1. / dP_drho
        kappaT = rho_kappaT / rhoad
        return kappaT

    def joule_thomson(self, params, alpha, rhoad, Tad, cp_id=2.5):
        out = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz)
        # out = self.d2helmholtz(params, alpha, rhoad, Tad)
        Ares, dAres_drho, dAres_dT, d2Ares_drho2, d2Ares_dT2, d2Ares_drho_dT = out

        # Isochoric heat capacity
        cv_res = - Tad * d2Ares_dT2

        # Pressure derivatives
        dP_dT_by_rhoad_res = rhoad * d2Ares_drho_dT
        dP_dT_by_rhoad_id = 1. 
        dP_dT_by_rhoad = dP_dT_by_rhoad_id +  dP_dT_by_rhoad_res 

        dP_drho_res = 2. * rhoad * dAres_drho + rhoad**2 * d2Ares_drho2
        dP_drho_id = 1. * Tad
        dP_drho = dP_drho_id + dP_drho_res

        # Thermal Expansion Coefficient
        alphap = dP_dT_by_rhoad / dP_drho

        # Isothermal Compressibility
        rho_kappaT = 1. / dP_drho

        # Residual Isobaric heat capacity
        cp_res = cv_res + Tad * alphap**2 / rho_kappaT - 1.
        # Total isobaric heat capacity
        cp = cp_res + cp_id

        # Joule Thomson coefficient
        muJT = (Tad * alphap - 1.) / (rhoad * cp)

        return muJT

    def thermophysical_properties(self, params, alpha, rhoad, Tad):

        # check that the shapes are correct
        assert rhoad.shape == Tad.shape
        assert alpha.shape == Tad.shape

        out = self.apply(params, alpha, rhoad, Tad, method=self.d2helmholtz)
        Ares, dAres_drho, dAres_dT, d2Ares_drho2, d2Ares_dT2, d2Ares_drho_dT = out

        # Compressibility Factor
        Pad_by_rhoad_res = rhoad * dAres_drho
        Pad_by_rhoad_id = 1. * Tad  # ideal gas Pad/rhoad
        Pad_by_rhoad = Pad_by_rhoad_id + Pad_by_rhoad_res
        Z = Pad_by_rhoad / Tad

        # Pressure calculation
        P = rhoad * Pad_by_rhoad

        # Internal Energy
        internal_res = Ares - Tad * dAres_dT
        internal_id = 1.5 * Tad
        internal = internal_id + internal_res

        # Isochoric Heat Capacity
        Cv_res = - Tad * d2Ares_dT2
        Cv_id = 1.5
        Cv = Cv_id + Cv_res

        # Pressure derivatives
        dP_dT_by_rhoad_res = rhoad * d2Ares_drho_dT
        dP_dT_by_rhoad_id = 1. 
        dP_dT_by_rhoad = dP_dT_by_rhoad_id +  dP_dT_by_rhoad_res 

        dP_drho_res = 2. * rhoad * dAres_drho + rhoad**2 * d2Ares_drho2
        dP_drho_id = 1. * Tad
        dP_drho = dP_drho_id + dP_drho_res

        # Thermal Expansion Coefficient
        alphap = dP_dT_by_rhoad / dP_drho

        # Isothermal Compressibility
        rho_kappaT = 1. / dP_drho
        kappaT = rho_kappaT / rhoad

        # Thermal Pressure Coefficient
        GammaV = rhoad * dP_dT_by_rhoad

        # Isobaric heat capacity
        Cp = Cv + Tad * alphap**2 / rho_kappaT

        # Adiabatic Index
        Gamma = Cp / Cv

        # Joule Thomson coefficient
        muJT = (Tad * alphap - 1.) / (rhoad * Cp)

        properties = {'alpha': alpha, 'temperature': Tad, 'density': rhoad,
                      'pressure': P,  'compressibility_factor': Z,
                      'internal_energy': internal,
                      'isochoric_heat_capacity': Cv,
                      'isothermal_compressibility': kappaT,
                      'rho_isothermal_compressibility': rho_kappaT,
                      'thermal_expansion_coefficient': alphap,
                      'thermal_pressure_coefficient': GammaV,
                      'isobaric_heat_capacity': Cp,
                      'adiabatic_index': Gamma,
                      'joule_thomson_coefficient': muJT}
        return properties


# Helper jitted functions that already have the model and params bounds
# useful for density solver, VLE and critical point solver
def helper_solver_funs(model, params):
    """
    Helper function to create jitted functions of the FE-ANN EoS
    These functions can be used for the density, VLE and critical point solver.

    The output functions are function only alpha, rhoad and Tad. The parameters
    are already bound to the function.

    Functions:
    helmholtz_fun: Helmholtz energy
    dhelmholtz_drho_fun: First derivative of the Helmholtz energy with respect to density
    d2helmholtz_drho2_dT_fun: Second derivative of the Helmholtz energy with respect to density and temperature
    d2helmholtz_drho2_fun: Second derivative of the Helmholtz energy with respect to density
    d2helmholtz_fun: Second derivative of the Helmholtz energy with respect to density and temperature

    pressure_fun: Pressure
    dpressure_drho_fun: First derivative of the pressure with respect to density
    d2pressure_drho2_fun: Second derivative of the pressure with respect to density
    dpressure_drho_aux_fun: First derivative of the pressure with respect to density auxiliary function
    pressure_and_chempot_fun: Pressure and chemical potential
    enthalpy_residual_fun: residual Enthalpy function
    thermal_expansion_coeff_fun: Thermal expansion coefficient function
    """
    helmholtz_fun = jax.jit(lambda alpha, rhoad, Tad: 
                            model.apply(params,
                                        jnp.atleast_1d(alpha),
                                        jnp.atleast_1d(rhoad),
                                        jnp.atleast_1d(Tad)))

    dhelmholtz_drho_fun = jax.jit(lambda alpha, rhoad, Tad:
                                   model.apply(params,
                                               jnp.atleast_1d(alpha),
                                               jnp.atleast_1d(rhoad),
                                               jnp.atleast_1d(Tad),
                                               method=model.dhelmholtz_drho)) 

    d2helmholtz_drho2_dT_fun = jax.jit(lambda alpha, rhoad, Tad:
                                        model.apply(params,
                                                    jnp.atleast_1d(alpha),
                                                    jnp.atleast_1d(rhoad),
                                                    jnp.atleast_1d(Tad),
                                                    method=model.d2helmholtz_drho2_dT)) 

    d2helmholtz_drho2_fun = jax.jit(lambda alpha, rhoad, Tad:
                                    model.apply(params,
                                                jnp.atleast_1d(alpha),
                                                jnp.atleast_1d(rhoad),
                                                jnp.atleast_1d(Tad),
                                                method=model.d2helmholtz_drho2)) 

    d2helmholtz_fun = jax.jit(lambda alpha, rhoad, Tad:
                              model.apply(params,
                                          jnp.atleast_1d(alpha),
                                          jnp.atleast_1d(rhoad),
                                          jnp.atleast_1d(Tad),
                                          method=model.d2helmholtz)) 


    pressure_fun = jax.jit(lambda alpha, rhoad, Tad: 
                           model.pressure(params,
                                          jnp.atleast_1d(alpha),
                                          jnp.atleast_1d(rhoad),
                                          jnp.atleast_1d(Tad)))

    dpressure_drho_fun = jax.jit(lambda alpha, rhoad, Tad: 
                                 model.dpressure_drho(params,
                                                      jnp.atleast_1d(alpha),
                                                      jnp.atleast_1d(rhoad),
                                                      jnp.atleast_1d(Tad)))

    d2pressure_drho2_fun = jax.jit(lambda alpha, rhoad, Tad: 
                                   model.d2pressure_drho2(params,
                                                          jnp.atleast_1d(alpha),
                                                          jnp.atleast_1d(rhoad),
                                                          jnp.atleast_1d(Tad)))

    dpressure_drho_aux_fun = jax.jit(lambda rhoad, alpha, Tad: 
                                     model.dpressure_drho(params,
                                                          jnp.atleast_1d(alpha),
                                                          jnp.atleast_1d(rhoad),
                                                          jnp.atleast_1d(Tad))[1])

    pressure_and_chempot_fun = jax.jit(lambda alpha, rhoad, Tad: 
                                        model.pressure_and_chempot(params,
                                                                   jnp.atleast_1d(alpha),
                                                                   jnp.atleast_1d(rhoad),
                                                                   jnp.atleast_1d(Tad)))

    enthalpy_residual_fun = jax.jit(lambda alpha, rhoad, Tad: 
                                    model.enthalpy_residual(params,
                                                            jnp.atleast_1d(alpha),
                                                            jnp.atleast_1d(rhoad),
                                                            jnp.atleast_1d(Tad)))

    thermal_expansion_coeff_fun = jax.jit(lambda alpha, rhoad, Tad:
                                          model.thermal_expansion_coeff(params,
                                                                        jnp.atleast_1d(alpha),
                                                                        jnp.atleast_1d(rhoad),
                                                                        jnp.atleast_1d(Tad)))

    thermophysical_properties_fun = jax.jit(lambda alpha, rhoad, Tad:
                                            model.thermophysical_properties(params,
                                                                        jnp.atleast_1d(alpha),
                                                                        jnp.atleast_1d(rhoad),
                                                                        jnp.atleast_1d(Tad)))

    # compiling functions
    ones_input = (jnp.ones(1), jnp.ones(1), jnp.ones(1))
    helmholtz_fun(*ones_input)
    dhelmholtz_drho_fun(*ones_input)
    d2helmholtz_drho2_dT_fun(*ones_input)
    d2helmholtz_drho2_fun(*ones_input)
    d2helmholtz_fun(*ones_input)

    pressure_fun(*ones_input)
    dpressure_drho_fun(*ones_input)
    d2pressure_drho2_fun(*ones_input)
    dpressure_drho_aux_fun(*ones_input)
    pressure_and_chempot_fun(*ones_input)
    enthalpy_residual_fun(*ones_input)
    thermal_expansion_coeff_fun(*ones_input)

    funs_dict = {'helmholtz_fun': helmholtz_fun,
                 'dhelmholtz_drho_fun': dhelmholtz_drho_fun,
                 'd2helmholtz_drho2_dT_fun': d2helmholtz_drho2_dT_fun,
                 'd2helmholtz_drho2_fun': d2helmholtz_drho2_fun,
                 'd2helmholtz_fun': d2helmholtz_fun,
                 'pressure_fun': pressure_fun,
                 'dpressure_drho_fun': dpressure_drho_fun, 
                 'd2pressure_drho2_fun': d2pressure_drho2_fun, 
                 'pressure_and_chempot_fun': pressure_and_chempot_fun,
                 'enthalpy_residual_fun': enthalpy_residual_fun,
                 'dpressure_drho_aux_fun': dpressure_drho_aux_fun,
                 'thermal_expansion_coeff_fun': thermal_expansion_coeff_fun,
                 'thermophysical_properties_fun': thermophysical_properties_fun}
    return funs_dict


# Helper function to get the alpha parameter
def helper_get_alpha(lambda_r, lambda_a):
    """
    Helper function to get the alpha parameter

    Parameters
    ----------
    lambda_r : float or array
        lambda_r parameter
    lambda_a : float or array
        lambda_a parameter

    Returns
    -------
    alpha : float or array
        alpha parameter
    """
    c_alpha = (lambda_r / (lambda_r-lambda_a)) * (lambda_r/lambda_a)**(lambda_a/(lambda_r-lambda_a))
    alpha = c_alpha*(1./(lambda_a-3) - 1./(lambda_r-3))
    return alpha
