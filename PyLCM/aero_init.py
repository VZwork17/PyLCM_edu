import numpy as np
from PyLCM.parameters import *
from PyLCM.micro_particle import *
from PyLCM.parcel import *
from PyLCM.condensation import *
from Post_process.print_plot import *

def model_init(
        dt, nt, do_condensation, do_collision, n_particles, \
        T_parcel, P_parcel, RH_parcel, w_parcel, z_parcel, max_z, \
        mode_aero_init, N_aero, mu_aero, sigma_aero, k_aero, \
        ascending_mode, display_mode, switch_kappa_koehler
    ):

    # RH to q conversion
    q_parcel    = RH_parcel * esatw( T_parcel ) / ( P_parcel - RH_parcel * esatw( T_parcel ) ) * r_a / rv

    # Truncate the array before taking the log if one of the N_aero_i is 0, which means that this mode will no longer be used.
    N_aero_array = np.array(N_aero) # First: Convert into np.array
    zeroindices  = np.where(N_aero_array==0) # Get the number of the mode which is empty
    zeroindices  = zeroindices[0]       # Conversion for better usage

    # Conversion of the other indices
    mu_aero_array = np.array(mu_aero)
    sigma_aero_array = np.array(sigma_aero)

    # Now delete the respective item in each array (N, mu, sigma)
    if len(zeroindices) > 0:
        # delete
        N_aero_array     = np.delete(N_aero_array, zeroindices)
        mu_aero_array    = np.delete(mu_aero_array, zeroindices)
        sigma_aero_array = np.delete(sigma_aero_array, zeroindices)

    # Now perform the log of mu and sigma arrays
    mu_aero_array = np.log(mu_aero_array)
    sigma_aero_array = np.log(sigma_aero_array)

    # Renaming of the arrrays
    N_aero = N_aero_array
    mu_aero = mu_aero_array
    sigma_aero = sigma_aero_array
    
    # Further initialization
    dz=0
    rho_parcel, V_parcel =  parcel_rho(P_parcel, T_parcel)
    wp_parcel = 0.0

    
    # Aerosol initialization
    T_parcel, q_parcel, particles_list = aero_init(mode_aero_init, n_particles, P_parcel,z_parcel, T_parcel,q_parcel, N_aero, mu_aero, sigma_aero, rho_aero, k_aero, switch_kappa_koehler)
    
    # Variant for mean radii and std. dev. of cloud+rain droplets only
    rc_liq_avg_array = np.zeros(nt+1)
    rc_liq_std_array = np.zeros(nt+1)

    # Optical thickness
    TAU_ts_array = np.zeros(nt+1)
    
    # Initialize array for all radii of particles for plot of their time evolution
    particles_array = np.zeros((nt+1, len(particles_list)))
    
    # Parcel routine
    # Initalize spectrum output
    spectra_arr = np.zeros((nt+1,len(rm_spec)))
    # Initialization of arrays for time series output
    qa_ts,qc_ts,qr_ts = np.zeros(nt+1),np.zeros(nt+1),np.zeros(nt+1)
    na_ts,nc_ts,nr_ts = np.zeros(nt+1),np.zeros(nt+1),np.zeros(nt+1)
    con_ts, act_ts, evp_ts, dea_ts = np.zeros(nt+1),np.zeros(nt+1),np.zeros(nt+1),np.zeros(nt+1)
    acc_ts, aut_ts, precip_ts = np.zeros(nt+1),np.zeros(nt+1), np.zeros(nt+1)
    spectra_arr[0],qa_ts[0], qc_ts[0],qr_ts[0], na_ts[0], nc_ts[0], nr_ts[0], particles_array[0], rc_liq_avg_array[0], rc_liq_std_array[0], TAU_ts_array[0] = ts_analysis(particles_list,air_mass_parcel,rm_spec, n_bins,n_particles, V_parcel, w_parcel, TAU_ts_array[0], dt, nt)
    
    # Initialization of arrays for T_parcel, RH_parcel, q_parcel and z_parcel. 
    # They will later be filled with values for each time step.
    T_parcel_array  = np.zeros(nt+1)
    RH_parcel_array = np.zeros(nt+1)
    q_parcel_array  = np.zeros(nt+1)
    z_parcel_array  = np.zeros(nt+1)
    wp_parcel_array = np.zeros(nt+1)
    P_parcel_array = np.zeros(nt+1)

    # Inserting the initialization values at the 0th position of the arrays.
    T_parcel_array[0]  = T_parcel
    RH_parcel_array[0] = (q_parcel * P_parcel / (q_parcel + r_a / rv)) / esatw( T_parcel ) 
    q_parcel_array[0]  = q_parcel
    z_parcel_array[0]  = z_parcel
    wp_parcel_array[0] = wp_parcel
    P_parcel_array[0] = P_parcel

    # Settings for the 'sine' and the 'in_cloud_oscillation' modes: time half wavelength of the parcel (s)
    time_half_wave_parcel = 600.0  # This value can be adapted by the user

    S_lst = 0.0
    
    return P_parcel, T_parcel, q_parcel, z_parcel, w_parcel, wp_parcel, N_aero, mu_aero, sigma_aero, nt, dt, max_z, do_condensation, do_collision, ascending_mode, time_half_wave_parcel, S_lst, display_mode, qa_ts, qc_ts, qr_ts, na_ts, nc_ts, nr_ts, T_parcel_array, P_parcel_array, RH_parcel_array, q_parcel_array, z_parcel_array, wp_parcel_array, particles_list, spectra_arr, con_ts, act_ts, evp_ts, dea_ts, acc_ts, aut_ts, precip_ts, particles_array, rc_liq_avg_array, rc_liq_std_array,n_particles, TAU_ts_array

def aero_init(mode_aero_init, n_ptcl, P_parcel, z_parcel,T_parcel,q_parcel, N_aero, mu_aero,sigma_aero,rho_aero, k_aero, switch_kappa_koehler):
    
    # Aerosol initialization
    
    particles_list = []
    # Computation of saturated water vapour pressure (e_s) and water vapour pressure of the parcel (e_a)
    e_s     = 611.2 * np.exp( 17.62 * ( T_parcel - 273.15 ) / ( T_parcel - 29.65 ) )
    e_a     = q_parcel * P_parcel / ( q_parcel + r_a / rv )
    # Computation of supersaturation
    S_adia  = ( e_a - e_s ) / e_s

    dql_liq = np.sum([p.M for p in particles_list])

    min_mass_aero = 1.0E-200 
    
    # Computation of aerosol inititial radius
    radius = np.logspace(np.log10(1.0E-9), np.log10(1.0E-6), n_ptcl)
    dlogr   = ( np.log(2.0E-6) - np.log(1.0E-9) ) / n_ptcl
    
    # Initialization of an array for n_particles of the 4 modes
    mode_count = len(N_aero)
    n_particles_mode_array = np.zeros(mode_count) 
    
    # Assigning the number of droplets to the 4 modes according to specified n_particles and N_aero[i]
    # (The result is now no longer integer!)
    n_particles_mode_array = n_ptcl * N_aero / np.sum(N_aero)

    # Truncate particle modes to integer and sum them up
    n_particles_mode_int = n_particles_mode_array.astype(int)
    n_difference = int(np.round(np.sum(n_particles_mode_array) - np.sum(n_particles_mode_int)))

    # Adjust the first mode to account for truncation
    n_particles_mode_int[0] += n_difference

    ## New initialization for the hygroscopicity parameter
    # Compute the indices for the modes
    mode_indices = np.cumsum(n_particles_mode_int)   
    
    # RNG (random number generator) method to generate distribution where all particles represent the same number of droplets
    if mode_aero_init == "Random":
        # Generate log-normal distribution for the modes
        temp_arr = []
        for k in range(mode_count):
            temp_arr.extend(np.random.lognormal(mu_aero[k], sigma_aero[k], n_particles_mode_int[k]))

        aero_r_seed = np.array(temp_arr)

        # Initialize particle (aerosol particles)
        for i in range(n_ptcl):
            particle = particles(i)
            
            if switch_kappa_koehler:
                # Determine the mode based on the index 'i' and set the appropriate kappa
                for idx in range(mode_count):
                    lower_bound = mode_indices[idx-1] if idx > 0 else 0
                    upper_bound = mode_indices[idx]

                    if lower_bound <= i < upper_bound:
                        particle.kappa = k_aero[idx]
                        break
                        
            particle.A = air_mass_parcel * np.sum(N_aero)/n_ptcl
            particle.Ns = aero_r_seed[i]**3 * 4./3. * np.pi * rho_aero * particle.A

            if particle.Ns > min_mass_aero:
                r_aero = (particle.Ns/ ( particle.A * 4.0 / 3.0 * pi * rho_aero ) )**(1.0/3.0)

                particle.M = max(r_aero, r_equi(S_adia,T_parcel,r_aero, rho_aero,switch_kappa_koehler,particle.kappa))**3 * particle.A * 4.0 / 3.0 * pi * rho_liq
            else: 
                particle.M = 0.0
            #Initalize particle position 
            particle.z = z_parcel
            particle.id = i
            #Put initialized particle in a particles_list 
            particles_list.append(particle)
            

    # Bin-like method to generate distribution where each particle represents different number of droplets
    elif mode_aero_init == "Weighting_factor":
        
        # Calculate the PDF (probability density function) of the overlapping log-normal distributions
        pdf_sum = np.zeros_like(radius)

        for N, mu, sigma in zip(N_aero, mu_aero, sigma_aero):
            pdf = lognormal_pdf(radius, mu,sigma)
            pdf_sum += N * pdf
            
        # Initialize particle (aerosol particles)
        for i in range(n_ptcl):
            particle = particles(i)
            # Define the range of values to evaluate the PDF
            particle.A = air_mass_parcel * pdf_sum[i] * dlogr * radius[i]
            particle.Ns = radius[i]**3 * 4./3. * np.pi * rho_aero * particle.A
            particle.kappa = 0.5

            if particle.Ns > min_mass_aero:
                r_aero = (particle.Ns/ ( particle.A * 4.0 / 3.0 * pi * rho_aero ) )**(1.0/3.0)
                particle.M = max(r_aero, r_equi(S_adia,T_parcel,r_aero, rho_aero,switch_kappa_koehler,particle.kappa))**3 * particle.A * 4.0 / 3.0 * pi * rho_liq
            else: 
                particle.M = 0.0
            
            #Initalize particle position 
            particle.z = z_parcel
            particle.id = i
            #Put initialized particle in a particles_list 
            particles_list.append(particle)

    dql_liq = (np.sum([p.M for p in particles_list]) - dql_liq)/air_mass_parcel
    T_parcel = T_parcel + dql_liq * l_v / cp
    q_parcel = q_parcel - dql_liq
    
    return(T_parcel, q_parcel, particles_list)


def lognormal_pdf(x, mu, sigma):
    """
    Calculates the log-normal PDF at a given x-value.
    
    Parameters:
        x (float): The value at which to calculate the PDF.
        mu (float): The mean of the underlying normal distribution.
        sigma (float): The standard deviation of the underlying normal distribution.
        
    Returns:
        float: The probability density at the given x-value.
    """
    coefficient = 1 / (x * sigma * np.sqrt(2 * np.pi))
    exponent = -((np.log(x) - mu)**2) / (2 * sigma**2)
    pdf_value = coefficient * np.exp(exponent)
    
    return pdf_value

def r_equi(S,T,r_aerosol, rho_aero, switch_kappa_koehler, kappa):
    # Limit supersaturation since higher saturations cause numerical issues.
    # Additionally, saturated or supersaturated conditions do not yield a unique solution.

    S_internal = min( S, -0.0001 ) # Higher saturations cause errors

    afactor = 2.0 * sigma_air_liq(T) / ( rho_liq * rv * T ) # Curvature effect
    if switch_kappa_koehler:
        bfactor = kappa
    else:
        bfactor = vanthoff_aero * rho_aero * molecular_weight_water / (rho_liq * molecular_weight_aero) # Solute effect

    # Iterative solver ( 0 = S - A / r + B / r^3 => r = ( B / ( A / r - S ) )^(1/3) )
    r_equi_0 = 1.0
    r_equi   = 1.0E-6

    while ( abs( ( r_equi - r_equi_0 ) / r_equi_0 ) > 1.0E-20 ):
        r_equi_0 = r_equi
        r_equi   = ( ( bfactor * r_aerosol**3 ) / ( afactor / r_equi - S_internal ) )**(1.0/3.0)
    return(r_equi)