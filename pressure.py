import numpy as np
import colossus
from colossus.cosmology import cosmology
from colossus.halo import concentration, mass_so, profile_nfw, mass_defs
from colossus.lss import peaks

beta_def = 1.0
eta_def = 0.7

G = colossus.utils.constants.G
cm_per_km = 1e5
km_per_kpc = colossus.utils.constants.KPC / cm_per_km  # KPC was in cm
s_per_Gyr = colossus.utils.constants.GYR
yr_per_Gyr = 1E9

#KS_model

def NFWf(x):
    return np.log(1. + x) - x/(1. + x)

# accepts radius in physical kpc/h


def NFWM(r, M, z, c, R):
    return M * NFWf(c*r/R) / NFWf(c)

# dissipation timescale


def t_d(r, M, z, c, R, beta=beta_def):
    Menc = NFWM(r, M, z, c, R)
    t_dyn = 2. * np.pi * (r**3 / (G*Menc))**(1./2.) * \
        km_per_kpc / (cosmology.getCurrent().H0 / 100.)
    return beta * t_dyn / s_per_Gyr / 2.

def Gamma(c_nfw):
    return 1.15 + 0.01*(c_nfw - 6.5)


def eta0(c_nfw):
    return 0.00676*(c_nfw - 6.5)**2 + 0.206*(c_nfw - 6.5) + 2.48


def NFWPhi(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    if(type(r) != np.ndarray and r == 0):
        return -1. * (G * M / R) * (c / NFWf(c))
    else:
        return -1. * (G * M / R) * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))

# this agrees with Komatsu and Seljak 2001 eqn 19 for theta:


def theta(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    # the rho0/P0 is actually 3eta^-1(0) * R/(GM) from Komatsu and Seljak 2001
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    return 1. + ((Gamma(c) - 1.) / Gamma(c))*rho0_by_P0*(NFWPhi(0, M, z, conc_model='diemer19', mass_def='vir')-NFWPhi(r, M, z, conc_model='diemer19', mass_def='vir'))

# arbitrary units for now while we figure out what to do with the normalization:


def rho_gas_unnorm(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    return theta(r, M, z, conc_model, mass_def)**(1.0 / (Gamma(c) - 1.0))

# in km/s:

# the complete sig2_tot that only makes one of each relevant function call:


def sig2_tot(r, M, z, conc_model='diemer19', mass_def='vir'):
    c = concentration.concentration(M, mass_def, z, model=conc_model)
    R = mass_so.M_to_R(M, z, mass_def)
    rho0_by_P0 = 3*eta0(c)**-1 * R/(G*M)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gamma(c) - 1.) / Gamma(c)) * \
        3. * eta0(c)**-1 * (phi0 - phir)
    return (1.0 / rho0_by_P0) * theta

# concentration increases with time
# radius also increases with time (hence why c increases)
def t_d(r, M, z, c, R, beta=beta_def):
    Menc = NFWM(r, M, z, c, R)
    t_dyn = 2. * np.pi * (r**3 / (G*Menc))**(1./2.) * \
        km_per_kpc / (cosmology.getCurrent().H0 / 100.)
    return beta * t_dyn / s_per_Gyr / 2.

# another possible definition for the dissipation timescale, Brunt-Vaisala timescale:

def dlnK_dlnr(r, M, z, c, R, fnth, gamma, beta):
    g = -G*M * NFWf(c*r/R) / (NFWf(c) * r**2)
    Gm = Gamma(c)
    phi0 = -1. * (c / NFWf(c))
    phir = -1. * (c / NFWf(c)) * (np.log(1. + c*r/R) / (c*r/R))
    theta = 1. + ((Gm - 1.) / Gm) * 3. * eta0(c)**-1 * (phi0 - phir)
    lnK = ((Gm - gamma) / (Gm - 1.) * np.log(theta)) +\
        np.log(1. - fnth)
    lnK_interp = interp(np.log(r), lnK)
    dlnK_dlnr = lnK_interp.derivative(1)
    return(dlnK_dlnr(np.log(r)))

#takes in Mobs, zobs, cosmo
# returns f_nth, sig2nth, sig2tot at z=zobs


def gen_fnth_shi(Mobs, zobs, cosmo,  mass_def='vir', conc_model='duffy08', beta=beta_def, eta=eta_def, nrads=30, zi=30., r_mult=1., timescale='td', init_eta=eta_def, conc_test_flag=False, psires=1e-4, dsig_pos=False, return_full=False, rads = None):
    data = mah_retriever(Mobs, zobs, cosmo)
    # This below is defunct, we switched to setting initial time based on m/M0
    #first_snap_to_use = np.where(data[:,0] <= zi)[0][0] - 1
    # first snap where mass is above psi_res
    first_snap_to_use = np.where(data[:, 1]/Mobs >= psires)[0][0]
    data = data[first_snap_to_use:]

    n_steps = data.shape[0] - 1

    Robs = mass_so.M_to_R(Mobs, zobs, mass_def)
    
    if rads is None:
        rads = np.logspace(np.log10(0.01*Robs), np.log10(r_mult*Robs), nrads)

    ds2dt = np.zeros((n_steps, nrads))
    sig2tots = np.zeros((n_steps, nrads))
    sig2nth = np.zeros((n_steps, nrads))

    # this process could be made more pythonic so that the looping is faster
    for i in range(0, n_steps):
        z_1 = data[i, 0]  # first redshift
        z_2 = data[i+1, 0]  # second redshift, the one we are actually at
        dt = cosmo.age(z_2) - cosmo.age(z_1)  # in Gyr
        mass_1 = data[i, 1]
        mass_2 = data[i+1, 1]
        dM = mass_2 - mass_1
        dMdt = data[i+1, 3]  # since the (i+1)th is computed between i+1 and i
        R_1 = mass_so.M_to_R(mass_1, z_1, mass_def)
        R_2 = mass_so.M_to_R(mass_2, z_2, mass_def)
        if(conc_model == 'vdb'):
            c_1 = data[i, 2]
            c_2 = data[i+1, 2]
        else:
            c_1 = concentration.concentration(
                mass_1, mass_def, z_1, model=conc_model)
            c_2 = concentration.concentration(
                mass_2, mass_def, z_2, model=conc_model)

        if(conc_test_flag):
            # set concentrations to 4 if t004 is further back than the furthest timestep we have data for
            # just to verify that results aren't affected
            t04_ind = np.where(data[:i+1, 1] > 0.04 * mass_2)[0][0]
            if(t04_ind == 0):
                print(i, z_1)
                c_1 = 4.
                c_2 = 4.

        sig2tots[i, :] = sig2_tot(rads, mass_2, c_2, R_2)  # second timestep
        if(i == 0):
            ds2dt[i, :] = (sig2tots[i, :] -
                           sig2_tot(rads, mass_1, c_1, R_1)) / dt
            sig2nth[i, :] = init_eta * sig2tots[i, :]
        else:
            ds2dt[i, :] = (sig2tots[i, :] - sig2tots[i-1, :]) / dt
            if(dsig_pos):
                # another check to make sure rare cases where dsig2dt is negative (turbulent energy removed)
                # doesn't affect results; it doesn't, since in general the halo should be growing in mass
                # and this should almost never happen
                ds2dt[i, ds2dt[i, :] < 0] = 0.
            if(timescale == 'td'):
                # t_d at z of interest z_2
                td = t_d(rads, mass_2, z_2, c_2, R_2, beta=beta_def)
            elif(timescale == 'tBV'):
                td = t_BV(rads, mass_2, z_2, c_2, R_2,
                          sig2nth[i-1, :] / sig2tots[i-1, :], 5./3., beta)
            sig2nth[i, :] = sig2nth[i-1] + \
                ((-1. * sig2nth[i-1, :] / td) + eta * ds2dt[i, :])*dt
            # can't have negative sigma^2_nth at any time
            sig2nth[i, sig2nth[i, :] < 0] = 0
    if(return_full == False):
        fnth = sig2nth[-1, :] / sig2tots[-1, :]
        return fnth, rads, sig2nth[-1, :], sig2tots[-1, :], data[-1, 0], data[-1, 2]
    else:
        fnth = sig2nth / sig2tots
        # return redshifts+concs too
        return fnth, rads, sig2nth[-1, :], sig2tots[-1, :], data[:, 0], data[:, 2]
    
    
def generate_nth_fraction(radius, mass, z, cosmo=None, model = 'Green20'):
    a, b, c, d, e, f = 0.495, 0.719, 1.417,-0.166, 0.265, -2.116
    if cosmo is None:
        print("No cosmology is set. Using Planck15 as default. Set cosmology with cosmology.setCosmology from colossus.")
        cosmo = cosmology.setCosmology('planck15')
    
    nuv = mass/peaks.peakHeight(mass, z)
    nth = a*(1+np.exp(-(radius/b)**c))*(nuv/4.1)**(d/(1+(radius/e)**f))
    if model=='Green20':
        return nth 
    elif model=='Aung22_TNG':
        f0 , M0, alpha, beta, ra = 0.0593, 1e14, 0.557, 0.448, 2.124
        nth_feedback = f0* (mass/M0)**(-alpha) * (1+z)**beta * (radius)**(ra)
        return nth + nth_feedback
    elif model=='Aung22_MGT':
        f0 , M0, alpha, beta, ra = 0.0421, 1e14, 0.514, 0.450, 2.236
        nth_feedback = f0* (mass/M0)**(-alpha) * (1+z)**beta * (radius)**(ra)
        return nth + nth_feedback
        
def pressure(radius, mass, r200m, z,  conc_model='diemer19', mass_def='200m', cosmo=None, model = 'Green20'):
    tot_pressure = rho_gas_unnorm(radius, mass, z, conc_model=conc_model, mass_def=mass_def)*\
    sig2_tot(radius, mass, z, conc_model=conc_model, mass_def=mass_def)
    fnth = generate_nth_fraction(radius/r200m, mass, z, cosmo = cosmo, model=model)
    return tot_pressure* (1-fnth)
    
