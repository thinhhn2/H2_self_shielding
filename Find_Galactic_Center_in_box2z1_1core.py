import yt
import numpy as np
from yt.data_objects.particle_filters import add_particle_filter
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.colors import LogNorm
import copy
import matplotlib
import astropy.units as u 
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from Find_Galactic_Center import Find_Galactic_Center
import sys

mode = sys.argv[1]
branch_idx = sys.argv[2]
start_idx = int(sys.argv[3])
end_idx = int(sys.argv[4])
use_previous_gal_com = True #if this is True, then the process cannot be parallelized
expand_factor = 2

if mode == 'apxc':
    rawtree = np.load('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_apx_corr/halotree_1434_final.npy', allow_pickle=True).tolist()
    pfs = np.loadtxt('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_apx_corr/pfs_allsnaps_1434.txt', dtype=str)[:,0]
elif mode == 'ray':
    rawtree = np.load('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_ray_temp/halotree_1434_final.npy', allow_pickle=True).tolist()
    pfs = np.loadtxt('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_ray_temp/pfs_allsnaps_1434.txt', dtype=str)[:,0]
elif mode == 'apx1':
    rawtree = np.load('/work/hdd/bdax/gtg115x/Halo_Finding//box_2_z_1_no-shield_run2/halotree_1434_final.npy', allow_pickle=True).tolist()
    pfs = np.loadtxt('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_no-shield_run2/pfs_allsnaps_1434.txt', dtype=str)[:,0]
elif mode == 'apx2':
    rawtree = np.load('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_no-shield_temp/halotree_1434_final.npy', allow_pickle=True).tolist()
    pfs = np.loadtxt('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_no-shield_temp/pfs_allsnaps_1434.txt', dtype=str)[:,0]

for idx in range(start_idx, end_idx, -1):
    ds = yt.load(pfs[int(idx)])
    if mode == 'apxc':
        metadata = np.load('/work/hdd/bdax/tnguyen2/H2-SelfShielding/stars_assignment/box_2_z_1_apx_corr/star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    elif mode == 'ray':
        metadata = np.load('/work/hdd/bdax/tnguyen2/H2-SelfShielding/stars_assignment/box_2_z_1_ray_temp/star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    elif mode == 'apx1':
        metadata = np.load('/work/hdd/bdax/tnguyen2/H2-SelfShielding/stars_assignment/box_2_z_1_no-shield_run2/star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    elif mode == 'apx2':
        metadata = np.load('/work/hdd/bdax/tnguyen2/H2-SelfShielding/stars_assignment/box_2_z_1_no-shield_temp/star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
    center = Find_Galactic_Center(ds = ds, oden = 2000, halo_center = rawtree[branch_idx][idx]['Halo_Center'],
                                  halo_rvir = expand_factor*rawtree[branch_idx][idx]['Halo_Radius'],
                                  star_pos = metadata['pos'], star_mass = metadata['mass'])
    if use_previous_gal_com == False:
        new_com, new_virRad = center.Find_Com_and_virRad()
    elif use_previous_gal_com == True:
        if mode == 'ray':
            previous_gal_com = np.load('/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_ray_temp/GalacticCenter_R2000/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (branch_idx, int(idx)+1), allow_pickle=True).tolist()['com']
        elif mode == 'apxc':
            previous_gal_com = np.load('/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_apx_corr/GalacticCenter_R2000/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (branch_idx, int(idx)+1), allow_pickle=True).tolist()['com']
        elif mode == 'apx1':
            previous_gal_com = np.load('/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_no-shield_run2/GalacticCenter_R2000/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (branch_idx, int(idx)+1), allow_pickle=True).tolist()['com']
        elif mode == 'apx2':
            previous_gal_com = np.load('/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_no-shield_temp/GalacticCenter_R2000/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (branch_idx, int(idx)+1), allow_pickle=True).tolist()['com']
        new_com, new_virRad = center.Find_Com_and_virRad(initial_gal_com_manual = True, initial_gal_com = (previous_gal_com*ds.units.code_length).to('m').v.tolist())
    #
    output_each = {}
    output_each['com'] = (new_com*center.ds.units.m).to('code_length').v
    output_each['r2000'] = (new_virRad*center.ds.units.m).to('code_length').v.tolist()
    if mode == 'apxc':
        np.save('/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_apx_corr/GalacticCenter_R2000/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (branch_idx, idx), output_each)
        savedir = '/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_apx_corr/GalacticCenter_R2000/GasProjection_Halo_%s_Snapshot_%s.png' % (branch_idx, idx)
    elif mode == 'ray':
        np.save('/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_ray_temp/GalacticCenter_R2000/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (branch_idx, idx), output_each)
        savedir = '/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_ray_temp/GalacticCenter_R2000/GasProjection_Halo_%s_Snapshot_%s.png' % (branch_idx, idx)
    elif mode == 'apx1':
        np.save('/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_no-shield_run2/GalacticCenter_R2000/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (branch_idx, idx), output_each)
        savedir = '/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_no-shield_run2/GalacticCenter_R2000/GasProjection_Halo_%s_Snapshot_%s.png' % (branch_idx, idx)
    elif mode == 'apx2':
        np.save('/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_no-shield_temp/GalacticCenter_R2000/Galaxy_Halo_%s_Snapshot_%s_comR2000.npy' % (branch_idx, idx), output_each)  
        savedir = '/work/hdd/bdax/tnguyen2/H2-SelfShielding/analysis/box_2_z_1_no-shield_temp/GalacticCenter_R2000/GasProjection_Halo_%s_Snapshot_%s.png' % (branch_idx, idx)
    center.plot_gas_projection(center = (new_com*center.ds.units.m).to('code_length').v, char_radius = (new_virRad*center.ds.units.m).to('code_length').v.tolist(), codelength_mode = False, saveplot=True, savedir=savedir)
    plt.close()
    del center, ds, metadata
    #center.plot_star_particles(center = (new_com*center.ds.units.m).to('code_length').v, radius = (new_virRad*center.ds.units.m).to('code_length').v.tolist(), saveplot=True, savedir=savedir)

    
    
