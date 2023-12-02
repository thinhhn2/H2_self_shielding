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
#Create yt filter for stars in general (Pop 2 + Pop 3)
def pop2(pfilter, data):
     filter_stars = np.logical_and(data["all", "particle_type"] == 7, data["all", "particle_mass"].to('Msun') > 1)
     return filter_stars

def pop3(pfilter, data):
     filter_stars = np.logical_and(data["all", "particle_type"] == 5, data["all", "particle_mass"].to('Msun') > 1)
     return filter_stars

def stars(pfilter, data):
     filter_pop2 = np.logical_and(data["all", "particle_type"] == 7, data["all", "particle_mass"].to('Msun') > 1)
     filter_pop3 = np.logical_and(data["all", "particle_type"] == 5, data["all", "particle_mass"].to('Msun') > 1)
     filter_stars = np.logical_or(filter_pop2,filter_pop3)
     return filter_stars

add_particle_filter("pop2", function=pop2, filtered_type="all", requires=["particle_type","particle_mass"])
add_particle_filter("pop3", function=pop3, filtered_type="all", requires=["particle_type","particle_mass"])
add_particle_filter("stars", function=stars, filtered_type="all", requires=["particle_type","particle_mass"])

def calculate_properties_Thinh(halo, sim_data, sfr_avetime = 0.005):
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
    coor = halo[0]
    rvir = halo[1]
    tree_loc = halo[2]
    
    #Get the redshift of the snapshot
    redshift = sim_data.current_redshift

    #Select the region of the halo
    reg = sim_data.sphere(coor,(rvir,"code_length"))

    #Calculate the gas mass in the region
    g_mass = reg[("gas","cell_mass")].to('Msun').sum().v.tolist()

    #Obtain the type of the particles to get the dark matter particle mass (type_particle == 1)
    type_particle = reg[('all','particle_type')]
    dm_mass = reg[("all","particle_mass")][np.logical_or(type_particle == 1,type_particle==4)].to('Msun').sum().v.tolist()

    #Calculate the metal mass
    metal_mass = reg[("gas","metal_mass")].to('Msun').sum().v.tolist()
    
    #Calculate the metallicity
    metallicity_each = reg[("gas","metallicity")]
    g_mass_each = reg[("gas","cell_mass")].to('Msun')
    metallicity = np.average(metallicity_each, weights = g_mass_each).v.tolist()

    #Create the star-type filter to make it easier to extract the creation time for the SFR calculation
    sim_data.add_particle_filter("stars")
    sim_data.add_particle_filter("pop2")
    sim_data.add_particle_filter("pop3")

    #Calculating total stellar mass
    sm_mass = reg["stars", "particle_mass"].in_units("Msun").sum().v.tolist()

    #Calculating the Pop2 and Pop3 stellar mass
    pop2_mass = reg["pop2", "particle_mass"].in_units("Msun").sum().v.tolist()
    pop3_mass = reg["pop3", "particle_mass"].in_units("Msun").sum().v.tolist()

    #Get the mass and the formation time for each star particle in the halo
    s_mass_each = reg["stars", "particle_mass"].in_units("Msun")
    formation_time = reg["stars", "creation_time"].in_units("Gyr")

    #Averaging the SFR 5 million years before the time of the snapshot
    sf_timescale = sfr_avetime*sim_data.units.Gyr
    currenttime = sim_data.current_time.in_units('Gyr')
    
    #SFR is in unit of Msun/yr
    sfr = s_mass_each[formation_time > currenttime - sf_timescale].sum().v.tolist()/sf_timescale.to('yr').tolist()

    #Calculate the mvir total mass from yt
    mvir = reg["all", "particle_mass"].in_units("Msun").sum().v.tolist() + g_mass

    #Calculate the gas mass fraction
    g_mass_fraction = g_mass/mvir

    #Calculate the H2 mass
    h2_mass = reg[("gas","H2_mass")].to('Msun').sum().tolist()
    
    #Calculate the weighted H2_fraction (the weight is the gas mass)
    h2_fraction_each = reg[("gas","H2_fraction")]
    h2_fraction = np.average(h2_fraction_each,weights=g_mass_each).v.tolist()

    #Make a dictionary for the output
    output_dict = {'tree_loc':tree_loc,'coor':coor,'Rvir':rvir,'redshift':redshift,'time':currenttime.v.tolist(),'gas_mass': g_mass, 'gas_mass_frac': g_mass_fraction,'dm_mass': dm_mass, 'star_mass': sm_mass,'pop2_mass': pop2_mass,'pop3_mass':pop3_mass ,'metal_mass': metal_mass, 'metallicity': metallicity,'sfr': sfr, 'total_mass':mvir,'h2_mass':h2_mass,'h2_fraction':h2_fraction}

    return output_dict

def add_additional_properties_Thinh(folder, hlist):
    gs = np.loadtxt('%s/pfs_manual.dat' % folder,dtype=str)
    #This represents the redshift. The index corresponds to the index in the pfs.dat file
    z_index = np.arange(0,len(gs),1)

    #Obtains the list of all the tree locations
    tree_loc_list = list(hlist.keys())

    #Sorting the merger_histories dictionary by redshift instead of tree branches to allow faster parallelization
    halo_by_z = {}
    for i in z_index:
        halos_each = []
        for mainkey, value in hlist.items():
            if i in value.keys():
                value_add = value[i]
                value_add.append(mainkey)
                halos_each.append(value_add)
        key_name = str(i)
        halo_by_z[key_name] = halos_each

    #Runnning parallel to add the halo properties
    my_storage = {}

    for sto, i in yt.parallel_objects(list(halo_by_z.keys()), nprocs-1, storage = my_storage):
        #The key in the halo_ns list is the index of the snapshot in the pfs.dat file (0 corresponds to DD0314, etc.)
        redshift_index = int(i)
        #Each processor obtains all the halos in each snapshot
        all_halos_z = halo_by_z[i]
        #Load the simulation
        sim_data = yt.load('%s/%s' % (folder,gs[redshift_index]))
        #Create an array to store the general halo information
        result_each_z = []
        #Run through the list of halo in one snapshot.
        for j in range(len(all_halos_z)):
            halo = all_halos_z[j]
            result_each_z.append(calculate_properties_Thinh(halo, sim_data)) #Obtain the halo's properties
        sto.result = {}
        sto.result[0] = i
        sto.result[1] = result_each_z

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

    return halo_by_loc

#--------------------------------------------------------------------------------------------------------------------------------------------------
#Main code

#The directory to the folder containing the simulation snapshots
folder = sys.argv[-1]
#The name of the halotree file
tree_name = 'halotree_trees0to5.npy'

output_name_Thinh = 'halotree_Thinh_structure.npy'

#Reload the merger history
hlist = np.load('%s/%s' % (folder,tree_name),allow_pickle=True).tolist()
#Adding more properties to the merger history
hlist_Thinh = add_additional_properties_Thinh(folder, hlist)
if yt.is_root():
    np.save('%s/%s' % (folder,output_name_Thinh),hlist_Thinh)