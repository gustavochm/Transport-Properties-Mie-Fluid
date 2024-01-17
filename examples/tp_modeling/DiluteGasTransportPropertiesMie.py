# import jax.numpy as np
import numpy as np


# Coefficients for collision integral Omega(1, 1)
# for Mie-6 potential
a_11 = np.zeros([6, 4])

a_11[0, 0] = 0.
a_11[0, 1] = -0.145269e1
a_11[0, 2] = 0.294682e2
a_11[0, 3] = 0.242508e1

a_11[1, 0] = 0.107782e-1
a_11[1, 1] = 0.587725
a_11[1, 2] = -0.180714e3
a_11[1, 3] = 0.595694e2

a_11[2, 0] = 0.546646e-1
a_11[2, 1] = -0.651465e1
a_11[2, 2] = 0.374457e3
a_11[2, 3] = -0.137807e3

a_11[3, 0] = 0.485352
a_11[3, 1] = 0.245523e2
a_11[3, 2] = -0.336782e3
a_11[3, 3] = 0.814187e2

a_11[4, 0] = -0.385355
a_11[4, 1] = -0.206868e2
a_11[4, 2] = 0.132246e3
a_11[4, 3] = 0.

a_11[5, 0] = 0.847232e-1
a_11[5, 1] = 0.521812e1
a_11[5, 2] = -0.181140e2
a_11[5, 3] = -0.747215e1


# Coefficients for collision integral Omega(2, 2)
# for Mie-6 potential
a_22 = np.zeros([6, 4])

a_22[0, 0] = 0.0
a_22[0, 1] = 0.113086e1
a_22[0, 2] = 0.234799e2
a_22[0, 3] = 0.310127e1

a_22[1, 0] = 0.0
a_22[1, 1] = 0.551559e1
a_22[1, 2] = -0.137023e3
a_22[1, 3] = 0.185848e2

a_22[2, 0] = 0.325909e-1
a_22[2, 1] = -0.292925e2
a_22[2, 2] = 0.243741e3
a_22[2, 3] = 0.0

a_22[3, 0] = 0.697682
a_22[3, 1] = 0.590192e2
a_22[3, 2] = -0.143670e3
a_22[3, 3] = -0.123518e3

a_22[4, 0] = -0.564238
a_22[4, 1] = -0.430549e2
a_22[4, 2] = 0.0
a_22[4, 3] = 0.137282e3

a_22[5, 0] = 0.126508
a_22[5, 1] = 0.104273e2
a_22[5, 2] = 0.150601e2
a_22[5, 3] = -0.408911e2


#####
def mie6_collision_integral(lambda_r, Tad, kk):

    if kk == '1,1':
        delta_kk = 0.0
        a_kk = a_11
    elif kk == '2,2':
        delta_kk = 1.0
        a_kk = a_22
    else:
        raise ValueError(f'kk: {kk} not defined, please use either "1,1" or "2,2" ')

    DELTA_kk = 0.5

    Tad_inv = 1. / Tad
    # lr_vector = np.array([1., 1/lambda_r, 1/lambda_r**2, 1/lambda_r**3]) # eq 4a
    lr_vector = np.array([1., 1/lambda_r, 1/lambda_r**2, 1/lambda_r**2*np.log(lambda_r)]) # eq 4b, with typo fixed

    ln_omega = -(2. / lambda_r) * np.log(Tad)
    ln_omega += delta_kk * np.log(1 - 2./(3*lambda_r))
    for i in range(1, 7):
        aii_kk = a_kk[i-1]
        ai_kk = np.dot(aii_kk, lr_vector)
        ln_omega += ( ai_kk * Tad_inv**((i-1)*DELTA_kk))

    omega = np.exp(ln_omega)
    return omega


def density_diffusivity_mie6_dilute(lambda_r, Tad):
    omega = mie6_collision_integral(lambda_r, Tad, kk="1,1")
    rhoD = (3. / (8. * omega)) * np.sqrt(Tad / np.pi)
    return rhoD


def viscosity_mie6_dilute(lambda_r, Tad):
    omega = mie6_collision_integral(lambda_r, Tad, kk="2,2")
    visc = (5. / (16. * omega)) * np.sqrt(Tad / np.pi)
    return visc


def thermal_conductivity_mie6_dilute(lambda_r, Tad):
    omega = mie6_collision_integral(lambda_r, Tad, kk="2,2")
    cv_id = 1.5
    tcond = (25. / (32. * omega)) * np.sqrt(Tad / np.pi) * cv_id
    return tcond
