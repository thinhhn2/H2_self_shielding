import numpy as np
import yt
import glob as glob
import sys,os
from yt.data_objects.particle_filters import add_particle_filter
import time
from yt.utilities.cosmology import Cosmology

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

#--------------------------------------------------------------------------------------------------------------------------------------------------
#This part is re-structuring the data system to fit Thinh's analysis
def calculate_properties_Thinh(halo, sim_data, sfr_avetime = 0.01):
    """
    Parameters
    ----------
    halo: list, the halo information from the dictionary
    sim_data: the loaded simulation
    sfr_avetime: the time to average the SFR. Default = 5 million years (0.005 Gyr)

    Returns
    -------
    A dictionary containing the properties of the input halo

    -------
    This function takes the halo information and return the gas and stellar properties
    of a halo

    """
    #Load the coordinates and radius of the halo (they are already in code_length unit)
    coor = halo['Halo_Center']
    rvir = halo['Halo_Radius']
    tree_loc = halo['tree_loc']
    #Get the redshift of the snapshot
    redshift = sim_data.current_redshift
    currenttime = sim_data.current_time.in_units('Gyr')
    #Select the region of the halo (using ds.r to avoid error from ds.sphere)
    reg = sim_data.r[(coor[0]-rvir):(coor[0]+rvir), (coor[1]-rvir):(coor[1]+rvir), (coor[2]-rvir):(coor[2]+rvir)]
    g_x_all = reg['gas','x'].to('code_length').v
    g_y_all = reg['gas','y'].to('code_length').v
    g_z_all = reg['gas','z'].to('code_length').v
    g_pos_all = np.vstack((g_x_all, g_y_all, g_z_all)).T
    p_pos_all = reg['all','particle_position'].to('code_length').v
    #Use boolean to reduce a box region (from ds.r) to a spherical region for a halo
    g_bool = np.linalg.norm(g_pos_all - coor, axis=1) < rvir
    p_bool = np.linalg.norm(p_pos_all - coor, axis=1) < rvir
    #Calculate the H2 mass
    h2_mass = reg[("gas","H2_mass")].to('Msun')[g_bool].sum().v.tolist()
    #Calculate the weighted H2_fraction (the weight is the gas mass)
    h2_fraction_each = reg[("gas","H2_fraction")][g_bool]
    g_mass_each = reg[("gas","cell_mass")].to('Msun')[g_bool]
    g_mass = g_mass_each.sum().v.tolist()
    if len(g_mass_each) != 0:
        h2_fraction = np.average(h2_fraction_each,weights=g_mass_each).v.tolist()
    else:
        h2_fraction = np.nan
    #Obtain the type of the particles to get the dark matter particle mass (type_particle == 1)
    ptype = reg[('all','particle_type')][p_bool]
    pmass_each = reg[('all','particle_mass')].to('Msun')[p_bool]
    dm_mass = pmass_each[np.logical_or(ptype == 1, ptype == 4)].to('Msun').sum().v.tolist()
    pop3_mass = pmass_each[np.logical_and(ptype == 5, pmass_each > 1)].to('Msun').sum().v.tolist()
    pop2_mass = pmass_each[np.logical_and(ptype == 7, pmass_each > 1)].to('Msun').sum().v.tolist()
    star_mass = pop2_mass + pop3_mass
    #Calculate the metal mass
    metal_mass = reg[("gas","metal_mass")].to('Msun')[g_bool].sum().v.tolist()
    #Calculate the metallicity
    metallicity_each = reg[("gas","metallicity")].to('Zsun')[g_bool]
    if len(g_mass_each) != 0:
        metallicity = np.average(metallicity_each, weights = g_mass_each).v.tolist()
    else:
        metallicity = np.nan
    #Calculate star formation rate
    star_boolean = np.logical_and(np.logical_or(ptype == 5, ptype == 7), pmass_each > 1)
    star_mass_each = pmass_each[star_boolean]
    formation_time = reg['all', 'creation_time'].to('Gyr')[p_bool][star_boolean]
    #Averaging the SFR 5 million years before the time of the snapshot
    sf_timescale = sfr_avetime*sim_data.units.Gyr
    #SFR is in unit of Msun/yr
    sfr = star_mass_each[formation_time > currenttime - sf_timescale].sum().v.tolist()/sf_timescale.to('yr').tolist()
    #Calculate the mvir total mass from yt
    mvir = pmass_each.sum().v.tolist() + g_mass
    #Calculate the gas mass fraction
    g_mass_fraction = g_mass/mvir
    #Make a dictionary for the output
    output_dict = {'tree_loc':tree_loc,'coor':coor,'Rvir':rvir,'redshift':redshift,'time':currenttime.v.tolist(),
                   'gas_mass': g_mass, 'gas_mass_frac': g_mass_fraction, 'h2_mass': h2_mass, 'h2_fraction': h2_fraction,
                   'dm_mass': dm_mass,'pop2_mass': pop2_mass,'pop3_mass': pop3_mass, 'star_mass': star_mass,
                   'metal_mass': metal_mass, 'metallicity': metallicity,'sfr': sfr, 'total_mass':mvir}
    return output_dict

def add_additional_properties_Thinh(folder, hlist):
    gs = np.loadtxt('%s/pfs_allsnaps_1013.txt' % folder,dtype=str)
    #This represents the redshift. The index corresponds to the index in the pfs.dat file
    z_index = np.arange(0,len(gs),1)
    #
    #Obtains the list of all the tree locations
    tree_loc_list = list(hlist.keys())
    #
    #Sorting the merger_histories dictionary by redshift instead of tree branches to allow faster parallelization
    halo_by_z = {}
    for i in z_index:
        halos_each = []
        for mainkey, value in hlist.items():
            if i in value.keys():
                value_add = value[i]
                value_add['tree_loc'] = mainkey
                halos_each.append(value_add)
        key_name = str(i)
        halo_by_z[key_name] = halos_each
    #
    #Runnning parallel to add the halo properties
    my_storage = {}
    #
    for sto, i in yt.parallel_objects(list(halo_by_z.keys()), nprocs, storage = my_storage):
        #The key in the halo_ns list is the index of the snapshot in the pfs.dat file (0 corresponds to DD0314, etc.)
        redshift_index = int(i)
        #Each processor obtains all the halos in each snapshot
        all_halos_z = halo_by_z[i]
        #Load the simulation
        sim_data = yt.load('%s' % gs[redshift_index])
        #Create an array to store the general halo information
        result_each_z = []
        #Run through the list of halo in one snapshot.
        for j in range(len(all_halos_z)):
            halo = all_halos_z[j]
            result_each_z.append(calculate_properties_Thinh(halo, sim_data)) #Obtain the halo's properties
        sto.result = {}
        sto.result[0] = i
        sto.result[1] = result_each_z
        print("Redshift index %s is finished" % redshift_index)
    #
    #Re-arrange the dictionary from redshift-sort to tree-location-sort
    halo_by_loc = {}
    for j in tree_loc_list:
        halos_each_loc = {}
        for c, vals in sorted(my_storage.items()):
            #loop through all the halos in one timestep
            for m in range(len(vals[1])):
                if j == vals[1][m]['tree_loc']:
                    halos_each_loc[vals[0]] = vals[1][m]
        halo_by_loc[j] = halos_each_loc
    #
    return halo_by_loc

#--------------------------------------------------------------------------------------------------------------
#The directory to the folder containing the simulation snapshots
folder = sys.argv[1]
halotree = sys.argv[2]
output_name_Thinh = sys.argv[3] #example 'halotree_Thinh_structure.npy'

#Reload the merger history
hlist = np.load('%s/%s' % (folder,halotree),allow_pickle=True).tolist()
#Adding more properties to the merger history
hlist_Thinh = add_additional_properties_Thinh(folder, hlist)
if yt.is_root():
    np.save('%s/%s' % (folder,output_name_Thinh),hlist_Thinh)

