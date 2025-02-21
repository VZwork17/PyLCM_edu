import numpy as np
import time

from PyLCM.parameters import *
from PyLCM.micro_particle import *
from PyLCM.aero_init import *
from PyLCM.parcel import *
from PyLCM.condensation import *
from PyLCM.collision import *
from PyLCM.animation import *
from PyLCM.widget import *

from Post_process.analysis import *
from Post_process.print_plot import *


# def timesteps_function(n_particles_widget, P_widget, RH_widget, T_widget, w_widget, nt_widget, dt_widget, rm_spec, ascending_mode_widget, mode_displaytype_widget, z_widget, max_z_widget, Condensation_widget, Collision_widget, mode_aero_init_widget, gridwidget, kohler_activation_radius, switch_kappa_koehler, switch_sedi_removal,entrainment_rate,switch_entrainment,qv_profiles, theta_profiles, entrainment_start, entrainment_end): 

def timesteps_function(
        dt, nt, do_condensation, do_collision, n_particles, \
        T_parcel, P_parcel, RH_parcel, w_parcel, z_parcel, max_z, \
        rm_spec, ascending_mode, display_mode, \
        mode_aero_init, N_aero, mu_aero, sigma_aero, k_aero, \
        kohler_activation_radius, switch_kappa_koehler, switch_sedi_removal, \
        entrainment_rate, switch_entrainment, entrainment_start, entrainment_end,
        qv_profiles, theta_profiles
    ):

    # # Limit the output by max z or max nt, whichever is smaller
    # time_to_top = (max_z_widget.value - z_widget.value) / w_widget.value
    # nt_to_top = time_to_top / dt_widget.value
    # if nt_to_top < nt_widget.value:
    #     nt_widget.value = nt_to_top
    
    # # Function call of the complete model initialization (model_init) (aerosol initialization included)
    # P_parcel, T_parcel, q_parcel, z_parcel, w_parcel, N_aero, mu_aero, sigma_aero, nt, dt, \
    # max_z, do_condensation, do_collision, ascending_mode, time_half_wave_parcel, S_lst, display_mode, \
    # qa_ts, qc_ts, qr_ts, na_ts, nc_ts, nr_ts, T_parcel_array, P_parcel_array, RH_parcel_array, q_parcel_array, \
    # z_parcel_array, particles_list, spectra_arr, con_ts, act_ts, evp_ts, dea_ts, acc_ts, aut_ts, precip_ts, particles_array, rc_liq_avg_array, rc_liq_std_array,n_particles, TAU_ts_array = model_init(dt_widget, nt_widget, Condensation_widget, Collision_widget, \
    #                             n_particles_widget, T_widget, P_widget, RH_widget, w_widget, z_widget, \
    #                             max_z_widget, mode_aero_init_widget, gridwidget, \
    #                             ascending_mode_widget, mode_displaytype_widget,switch_kappa_koehler)

    # Limit the output by max z or max nt, whichever is smaller
    time_to_top = (max_z - z_parcel) / w_parcel
    nt_to_top = time_to_top / dt
    if nt_to_top < nt:
        nt = nt_to_top

    # Function call of the complete model initialization (model_init) (aerosol initialization included)
    P_parcel, T_parcel, q_parcel, z_parcel, w_parcel, wp_parcel, N_aero, mu_aero, sigma_aero, nt, dt, \
    max_z, do_condensation, do_collision, ascending_mode, time_half_wave_parcel, S_lst, display_mode, \
    qa_ts, qc_ts, qr_ts, na_ts, nc_ts, nr_ts, T_parcel_array, P_parcel_array, RH_parcel_array, q_parcel_array, \
    z_parcel_array, wp_parcel_array, particles_list, spectra_arr, con_ts, act_ts, evp_ts, dea_ts, acc_ts, aut_ts, precip_ts, particles_array, rc_liq_avg_array, rc_liq_std_array,n_particles, TAU_ts_array = \
        model_init(
            dt, int(nt), do_condensation, do_collision, n_particles, \
            T_parcel, P_parcel, RH_parcel, w_parcel, z_parcel, max_z, \
            mode_aero_init, N_aero, mu_aero, sigma_aero, k_aero, \
            ascending_mode, display_mode, switch_kappa_koehler
        )


################################
    # Timestep routine
################################

    # Create array for the drop radii evolution

    if display_mode == 'graphics':
        # Initialization of animation
        figure_item = animation_init(dt, nt,rm_spec, qa_ts, qc_ts, qr_ts, na_ts, nc_ts, nr_ts, T_parcel_array, RH_parcel_array, q_parcel_array, z_parcel_array)

    for t in range(nt):

        time = (t+1)*dt

        # Parcel ascending
        z_parcel, T_parcel,P_parcel, wp_parcel = ascend_parcel(z_parcel, T_parcel,P_parcel, w_parcel, wp_parcel, dt, time,max_z,theta_profiles, time_half_wave_parcel, ascending_mode)
        
        if switch_entrainment and (entrainment_start <= time) and  (time < entrainment_end) and (z_parcel < 3000.):
            #Entrainment works only when z < 3000m
            T_parcel, q_parcel = basic_entrainment(dt,z_parcel, T_parcel, q_parcel,P_parcel, entrainment_rate, qv_profiles, theta_profiles)
    
        #rho_parcel, V_parcel, air_mass_parcel =  parcel_rho(P_parcel, T_parcel)
        rho_parcel, V_parcel =  parcel_rho(P_parcel, T_parcel)

        # Check for and filter out particles of 0 mass/weighting
        particles_list = [particle for particle in particles_list if particle.M != 0 and particle.A != 0]
        
        # Condensational Growth
        dq_liq = 0.0
        if do_condensation:
            particles_list, T_parcel, q_parcel, S_lst, con_ts[t+1], act_ts[t+1], evp_ts[t+1], dea_ts[t+1] = drop_condensation(particles_list, T_parcel, q_parcel, P_parcel, nt, dt, air_mass_parcel, S_lst, rho_aero,kohler_activation_radius, con_ts[t+1], act_ts[t+1], evp_ts[t+1], dea_ts[t+1],switch_kappa_koehler)
            
            # Convert mass output to per mass per sec.
            con_ts[t+1]  = 1e3 * con_ts[t+1] / air_mass_parcel / dt
            act_ts[t+1]  = 1e3 * act_ts[t+1] / air_mass_parcel / dt
            evp_ts[t+1]  = 1e3 * evp_ts[t+1] / air_mass_parcel / dt
            dea_ts[t+1]  = 1e3 * dea_ts[t+1] / air_mass_parcel / dt

        # Check for and filter out particles of 0 mass/weighting
        particles_list = [particle for particle in particles_list if particle.M != 0 and particle.A != 0]

        # Collisional Growth
        if do_collision:

            particles_list, acc_ts[t+1], aut_ts[t+1],precip_ts[t+1] = collection(dt, particles_list,rho_parcel, rho_liq, P_parcel, T_parcel, acc_ts[t+1], aut_ts[t+1],precip_ts[t+1], switch_sedi_removal, z_parcel, max_z, w_parcel)
            
            # Convert mass output to per mass per sec.
            acc_ts[t+1]  = 1e3 * acc_ts[t+1] / air_mass_parcel / dt
            aut_ts[t+1]  = 1e3 * aut_ts[t+1] / air_mass_parcel / dt

        # Check for and filter out particles of 0 mass/weighting
        particles_list = [particle for particle in particles_list if particle.M != 0 and particle.A != 0]

        # Analysis
        spectra_arr[t+1],qa_ts[t+1], qc_ts[t+1],qr_ts[t+1], na_ts[t+1], nc_ts[t+1], nr_ts[t+1], particles_array[t+1], rc_liq_avg_array[t+1], rc_liq_std_array[t+1], TAU_ts_array[t+1] = ts_analysis(particles_list,air_mass_parcel,rm_spec, n_bins, n_particles, V_parcel, w_parcel, TAU_ts_array[t], dt, nt)
        RH_parcel = (q_parcel * P_parcel / (q_parcel + r_a / rv)) / esatw( T_parcel ) 
        
        # Saving values of T_parcel, RH_parcel, q_parcel, z_parcel for every timestep (needed for plots)
        T_parcel_array[t+1]  = T_parcel
        RH_parcel_array[t+1] = RH_parcel
        q_parcel_array[t+1]  = q_parcel
        z_parcel_array[t+1]  = z_parcel
        wp_parcel_array[t+1]  = wp_parcel

        P_parcel_array[t+1] = P_parcel
                
        time_array = np.arange(nt+1)*dt

        # Display of variables during runtime
        if display_mode == 'text_fast':
            # Prints text output at every second
            if (time%1) ==0:
                print_output(t,dt, z_parcel, T_parcel, q_parcel, RH_parcel, qc_ts[t+1], qr_ts[t+1], na_ts[t+1], nc_ts[t+1], nr_ts[t+1], particles_list)
        elif display_mode == 'graphics':
            # Displays and continuously updates plots during runtime using plotly
            # Figure output is updated every 5 seconds
            if (time%5) == 0:
                animation_call(figure_item, time_array, t, dt, nt,rm_spec, qa_ts, qc_ts, qr_ts, na_ts, nc_ts, nr_ts, T_parcel_array, RH_parcel_array, q_parcel_array, z_parcel_array)
        
        if z_parcel == max_z:
            break
    
    
    # Calculate albedo from tau
    gamma = 7.7 # 13.3
    albedo_array = TAU_ts_array / ( gamma + TAU_ts_array )

    return nt, dt, time_array, T_parcel_array, P_parcel_array, RH_parcel_array, q_parcel_array, z_parcel_array, wp_parcel_array, qa_ts,qc_ts,qr_ts, na_ts, nc_ts, nr_ts, spectra_arr, con_ts, act_ts, evp_ts, dea_ts, acc_ts, aut_ts, precip_ts, particles_array, rc_liq_avg_array, rc_liq_std_array, TAU_ts_array, albedo_array
    
