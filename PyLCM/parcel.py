import numpy as np
import random
import matplotlib.pyplot as plt
from PyLCM.parameters import *
from PyLCM.micro_particle import *
from PyLCM.condensation import *
from PyLCM.entrainment import *

random.seed(10)

def parcel_rho(P_parcel, T_parcel):
    # Computes density and volume of the parcel for given pressure and temperature.
    rho_parcel = P_parcel / ( r_a * T_parcel ) #  Air density
    V_parcel   = air_mass_parcel / rho_parcel # (Assumed) volume of parcel for a 100 kg air parcel
    
    return(rho_parcel, V_parcel)

def ascend_parcel(z_parcel, T_parcel,P_parcel,w_parcel, wp_parcel, dt, time, max_z,theta_profiles,time_half_wave_parcel=1200.0, ascending_mode='linear', t_start_oscillation=800, noise_amplitude=.05):
    # Computes values for the ascending parcel. Three ascending mode options are provided.
    # Users can change the half wavelength of the oscillation (time_half_wave_parcel (s)) and the oscillation start time (t_start_oscillation (s), only relevant for the 'in_cloud_oscillation' case)
    if ascending_mode=='linear':
        # Linear ascending
        if z_parcel < max_z: 
            dz = w_parcel * dt
            z_parcel   = z_parcel + dz
            T_parcel   = T_parcel - dz * g / cp
        #change environmental pressure
            theta_env  = get_interp1d_var(z_parcel,z_env,theta_profiles)
            T_env      = theta_env * (P_parcel / p0) ** (r_a / cp)
            P_parcel   = P_parcel - P_parcel * g * dz / ( r_a * T_env )
    elif ascending_mode=='sine': 
        # Sinusoidal oscillation
        w_oscillate = w_parcel * np.pi / 2.0 * np.sin(np.pi * time / time_half_wave_parcel)
        dz = w_oscillate  * dt
        z_parcel = z_parcel + dz
        if w_parcel > 0:
            T_parcel = T_parcel - dz * g / cp
        else:
            T_parcel = T_parcel + dz * g / cp
            
        #change environmental pressure
        theta_env  = get_interp1d_var(z_parcel,z_env,theta_profiles)
        T_env      = theta_env * (P_parcel / p0) ** (r_a / cp)
        P_parcel   = P_parcel - P_parcel * g * dz / ( r_a * T_env )

    elif ascending_mode=='in_cloud_oscillation':
        # The particle rises first linearly. After oscillation start time it starts to oscillate.
        phase = np.arccos(2/np.pi)
        if time < t_start_oscillation:
            dz = w_parcel * dt
            z_parcel   = z_parcel + dz
            T_parcel   = T_parcel - dz * g / cp
        else:
            w_oscillate = w_parcel * np.pi / 2.0 * np.cos(np.pi * (time-t_start_oscillation) / time_half_wave_parcel + phase)
            dz = w_oscillate  * dt
            z_parcel = z_parcel + dz
            if w_parcel > 0:
                T_parcel = T_parcel - dz * g / cp
            else:
                T_parcel = T_parcel + dz * g / cp

        #change environmental pressure
        theta_env  = get_interp1d_var(z_parcel,z_env,theta_profiles)
        T_env      = theta_env * (P_parcel / p0) ** (r_a / cp)
        P_parcel   = P_parcel - P_parcel * g * dz / ( r_a * T_env )

    elif ascending_mode=='turbulent':
        # The vertical velocity is subject to AR(1) process.
        alpha_w = np.exp( - dt / tau_corr )

        wp_parcel = alpha_w * wp_parcel + np.sqrt( 1.0 - alpha_w ** 2 ) * noise_amplitude * random.gauss(0,1)
        w_turb = w_parcel + wp_parcel

        dz = w_turb * dt
        z_parcel = z_parcel + dz
        T_parcel   = T_parcel - dz * g / cp

        #change environmental pressure
        theta_env  = get_interp1d_var(z_parcel,z_env,theta_profiles)
        T_env      = theta_env * (P_parcel / p0) ** (r_a / cp)
        P_parcel   = P_parcel - P_parcel * g * dz / ( r_a * T_env )


    return z_parcel, T_parcel, P_parcel, wp_parcel

#Functions to make environmental profiles for three different stability conditions
def create_env_profiles(T_init, qv_init,z_init,p_env, stability_condition):
    #Create temperature and vapor content profiles based on initial conditions.
    z_env = np.arange(z_init, 3001, 10) # vertical levels up to 3000m
    if stability_condition == 'Stable':
        lapse_rates = 5 / 1000 # -6.5 K/km converted to K/m
    elif stability_condition == 'Neutral':
        lapse_rates = 0           # 0 K/km
    elif stability_condition == 'Unstable':
        lapse_rates = -6.5 / 1000    # 5 K/km converted to K/m
    else:
        raise ValueError(f"Unknown stability condition: {stability_condition}")
        
    qv_diff = (qv_init - 2*1e-3) / len(z_env)# Create linear qv profile down to 2 g/kg at the boudary layer top
    
    qv_profiles = np.maximum(qv_init - qv_diff * np.arange(len(z_env)), 2*1e-3)
    
    theta_init = T_init * ( p0 / p_env )**( r_a / cp )
    
    theta_profiles = theta_init + lapse_rates * z_env
        
    fig, ax1 = plt.subplots(figsize=(4, 6))
    ax1.plot(theta_profiles, z_env, c="r", lw=3, label=r"$ \Theta $ (K)")
    ax1.set_xlabel(r"$ \Theta $ (K)", color='r')
    ax1.tick_params(axis='x', colors='r') 
    
    # Create a second axis that shares the same y-axis
    ax2 = ax1.twiny()
    ax2.plot(qv_profiles*1e3, z_env, c="k", ls="--", lw=3, label=r"$q_{\mathrm{v}}$ (g/kg)")
    ax2.set_xlabel(r"$q_{\mathrm{v}}$ (g/kg)")
    ax2.tick_params(axis='x', colors='k')  
    
    plt.ylabel("z (m)")
    # Add a legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)

    plt.title(stability_condition + " condition")
    plt.show()

    return qv_profiles, theta_profiles, z_env
