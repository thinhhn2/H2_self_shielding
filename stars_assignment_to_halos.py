#VERSION 15: USE 'R200' (OR CLOSEST VALUE TO IT) INSTEAD OF 'HALO_RADIUS' 

import numpy as np
import yt
from astropy.constants import G
import astropy.units as u
import os
import glob as glob
import healpy as hp
from scipy.spatial.distance import cdist
from scipy.signal import savgol_filter
import time as time_sys
import collections
import sys

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size


def search_closest_upper(value, array):
    diff = array - value
    return np.where(diff >= 0)[0][0]

def extract_and_order_snapshotIdx(rawtree, branch):
    #this function extract only the snapshot key (i.e. the integer value) from the rawtree halotree output
    keys = list(rawtree[branch].keys())
    snapshotIdx = [x for x in keys if not isinstance(x, str)]
    snapshotIdx.sort()
    return snapshotIdx

def get_r200_radius(rawtree, branch, idx):
    #this function return r200 value or find the closest value to it (in case the halo does not have 'r200' radius)
    if 'r200' in rawtree[branch][idx].keys():
        return rawtree[branch][idx]['r200']
    else:
        key_list = list(rawtree[branch][idx].keys())
        r_keys = np.array([x[1:] for x in key_list if x[0] =='r'])
        r_key = r_keys[abs(r_keys.astype(float)-200)==abs(r_keys.astype(float)-200).min()][0]
        return rawtree[branch][idx]['r'+r_key]

def list_of_halos_wstars_idx(rawtree, pos_allstars, idx):
    halo_wstars_pos = np.empty(shape=(0,3))
    halo_wstars_rvir = np.array([])
    halo_wstars_branch = np.array([])
    for branch, vals in rawtree.items():
        if idx in vals.keys():
            if (np.linalg.norm(pos_allstars - rawtree[branch][idx]['Halo_Center'], axis=1) < get_r200_radius(rawtree, branch, idx)).any():
                halo_wstars_pos = np.vstack((halo_wstars_pos, vals[idx]['Halo_Center']))
                halo_wstars_rvir = np.append(halo_wstars_rvir, get_r200_radius(rawtree, branch, idx))
                halo_wstars_branch = np.append(halo_wstars_branch, branch)   
    return halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch

def vecs_calc(nside):
    pix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside,np.arange(pix))
    vecs = hp.ang2vec(np.array(theta),np.array(phi))
    return vecs

def cut_particles(pos,mass,center,cut_size=700,dense=False,segments=1,timing=0):
    if timing:
        time5 = time_sys.time()
    bool = np.full(len(pos),True)
    vec = {}
    if not dense:
        vec[0] = vecs_calc(1)
        vec[1] = vecs_calc(1)
        vec[2] = vecs_calc(1)
    else:
        vec[0] = vecs_calc(1)
        vec[1] = vecs_calc(2)
        vec[2] = vecs_calc(2)
    dist_pos = np.linalg.norm(pos-center,axis=1)
    inner = np.array([40,20,12])
    annuli = np.linspace(0,dist_pos.max()/3,inner[0])
    annuli2 = np.linspace(dist_pos.max()/3,2*dist_pos.max()/3,inner[1])[1:]
    annuli3 = np.linspace(2*dist_pos.max()/3,dist_pos.max(),inner[2])[1:]
    annuli = np.append(annuli,annuli2)
    annuli = np.append(annuli,annuli3)
    index = np.arange(len(pos))
    for i in range(len(annuli)-1):
        bool_in_0 = (dist_pos <= annuli[i+1])*(dist_pos > annuli[i])
        cutlength = cut_size*annuli[i+1]/annuli[-1]
        current_group = np.arange(len(inner))[(i >= np.cumsum(inner)-inner)*(i < np.cumsum(inner))][0]
        pos_norm = dist_pos[bool_in_0][:,np.newaxis]
        pos_in = pos[bool_in_0]
        vec_ang = np.dot((pos_in-center),vec[current_group].T)
        if len(np.unique(vec_ang)) != len(vec_ang):
            vec_ang += vec_ang.min()*1e-5*np.random.random(len(vec_ang[0]))[np.newaxis,:]
        #pos_group = np.searchsorted(vec_ang,vec_ang.max(axis=1))
        pos_group = np.where(vec_ang == vec_ang.max(axis=1)[:,np.newaxis])[1]
        # lenpos = len(pos_group)
        # lenind = len(index[bool_in_0])
        # if lenpos > lenind:
        #     print('Mismatch',lenpos,lenind,vec_ang.shape,pos_in.shape,len(bool_in_0))
        #     print(np.where(vec_ang == vec_ang.max(axis=1)[:,np.newaxis])[1].shape)
        #     pos_group = pos_group[:-(lenpos-lenind)]
        for t in range(len(vec[current_group])):
            current_index = index[bool_in_0][pos_group==t]
            if len(current_index) > max(cutlength,max(10/segments,1)):
                mass_tot = mass[current_index].sum()
                cut = int(np.ceil(len(current_index)/cutlength))
                bool[current_index] = False
                rand_ind = np.random.choice(current_index,size=len(current_index),replace=False)
                bool[rand_ind[0::cut]] = True
                mass_in = mass[rand_ind[0::cut]].sum()
                mass[rand_ind[0::cut]] *= mass_tot/mass_in
        if timing and time_sys.time()-time5 >timing:
            print('Make Annuli',time_sys.time()-time5)
            time5 = time_sys.time()
    del pos
    mass[index[np.logical_not(bool)]] *= 1e-10
    return mass[bool], bool

def find_total_E(star_pos, star_vel, ds, rawtree, branch, idx):
    #
    #This function finds the total energy of an array of star particles in one halo at a certain timestep.
    if star_pos.shape == (3,): #reshaping the star_pos and star_vel to be 2D arrays, in the case of a single star
        star_pos = star_pos.reshape(1,3)
        star_vel = star_vel.reshape(1,3)
    #
    regA = ds.sphere(rawtree[branch][idx]['Halo_Center'], get_r200_radius(rawtree, branch, idx))
    #
    massA = regA['all','particle_mass'].to('kg')
    posA = regA['all','particle_position'].to('m')
    #
    boolmass = massA.to('Msun') > 1
    boolloc = np.linalg.norm(posA.to('code_length').v - rawtree[branch][idx]['Halo_Center'], axis=1) <= get_r200_radius(rawtree, branch, idx)
    boolall = boolmass*boolloc
    #
    posA = posA[boolall]
    massA = massA[boolall]
    #
    if len(massA) > 10000:
        centerA = (rawtree[branch][idx]['Halo_Center']*ds.units.code_length).to('m')
        massA_cut, boolA_cut = cut_particles(posA.v,massA.v,centerA.v)
        posA_cut = posA[boolA_cut]
    else:
        massA_cut = massA.to('kg').v
        posA_cut = posA
    del posA, massA
    #
    #use cdist, 100x faster
    disAinv_cut = 1/cdist((star_pos*ds.units.code_length).to('m').v, posA_cut.v, 'euclidean')
    disAinv_cut[~np.isfinite(disAinv_cut)] = 0
    disAinv_cut[np.isnan(disAinv_cut)] = 0
    #
    PE = np.sum(-G.value*massA_cut*disAinv_cut, axis=1)
    velcom = (rawtree[branch][idx]['Vel_Com']*ds.units.code_length/ds.units.s).to('m/s').v
    KE = 0.5*np.linalg.norm(star_vel - velcom, axis=1)**2
    E = KE + PE
    E[np.isnan(E)] = 1e99
    return E

def region_number(idx, halo_dir):
    #this function find the refined region that will be used for a given snapshot
    lenregion = len(halo_dir + '/' + 'refined_region_')
    regions = glob.glob(halo_dir + '/' + 'refined_region_*.npy')
    region_list = []
    for region in regions:
        region_list.append(int(region[lenregion:-4]))
    region_list.sort()
    region_list = np.array(region_list)
    region_idx = region_list[region_list >= idx].min()
    return region_idx

def extract_star_metadata(pfs, idx, numsegs, halo_dir, metadata_dir):
    #
    ds = yt.load(pfs[idx])
    #load the refined region, we assume that stars only exist in this region
    #Combine all the refined regions from all timesteps 
    ll_all,ur_all = np.array([1e99,1e99,1e99]),-1*np.array([1e99,1e99,1e99])
    refined_files = glob.glob(halo_dir + '/' + 'refined_region_*.npy')
    for file in refined_files:
        ll_o,ur_o = np.load(file,allow_pickle=True).tolist()
        ll_all = np.minimum(ll_all,np.array(ll_o))
        ur_all = np.maximum(ur_all,np.array(ur_o))
    buffer_all = (ur_all - ll_all)*0.05
    ll_all, ur_all = ll_all - buffer_all, ur_all + buffer_all
    #
    xx,yy,zz = np.meshgrid(np.linspace(ll_all[0],ur_all[0],numsegs),\
                np.linspace(ll_all[1],ur_all[1],numsegs),np.linspace(ll_all[2],ur_all[2],numsegs))

    _,segdist = np.linspace(ll_all[0],ur_all[0],numsegs,retstep=True)

    ll = np.concatenate((xx[:-1,:-1,:-1,np.newaxis],yy[:-1,:-1,:-1,np.newaxis],zz[:-1,:-1,:-1,np.newaxis]),axis=3) #ll is lowerleft
    ur = np.concatenate((xx[1:,1:,1:,np.newaxis],yy[1:,1:,1:,np.newaxis],zz[1:,1:,1:,np.newaxis]),axis=3) #ur is upperright
    ll = np.reshape(ll,(ll.shape[0]*ll.shape[1]*ll.shape[2],3))
    ur = np.reshape(ur,(ur.shape[0]*ur.shape[1]*ur.shape[2],3))
    #Runnning parallel to load the star information
    my_storage = {}
    for sto, i in yt.parallel_objects(range(len(ll)), nprocs, storage = my_storage, dynamic = False):
        buffer = (segdist/200) #set buffer when loading the box
        reg = ds.box(ll[i] - buffer ,ur[i] + buffer)
        ptype = reg['all', 'particle_type'].v
        pmass = reg['all', 'particle_mass'].to('Msun').v
        ppos = reg['all', 'particle_position'].to('code_length').v
        pvel = reg['all', 'particle_velocity'].to('km/s').v
        page = reg['all','age'].to('Gyr').v
        #pmet = reg['all','metallicity_fraction'].to('Zsun').v
        pID = reg['all','particle_index'].v
        #
        star_bool = np.logical_and(np.logical_or(ptype == 5, ptype == 7), pmass > 1)
        sto.result = {}
        sto.result['type'] = ptype[star_bool]
        sto.result['mass'] = pmass[star_bool]
        sto.result['pos'] = ppos[star_bool]
        sto.result['vel'] = pvel[star_bool]
        sto.result['age'] = page[star_bool]
        #sto.result['met'] = pmet[star_bool]
        sto.result['ID'] = pID[star_bool]
    #
    output = {}
    infos = ['type', 'mass', 'pos', 'vel', 'age', 'ID']
    for info in infos:
        if info == 'pos' or info == 'vel':
            output[info] = np.empty(shape=(0,3))
        else:
            output[info] = np.array([])
    #Go through different cores/regions to load the stars info to the output
    for c, vals in sorted(my_storage.items()):
        for info in infos:
            if info == 'pos' or info == 'vel':
                output[info] = np.vstack((output[info], vals[info]))
            else:
                output[info] = np.append(output[info], vals[info])
    #Because there is a buffer region, we use np.unique to remove the duplicated stars
    if yt.is_root():
        #print('Before removing duplicates, the number of stars is:', len(output['ID']))
        unique_idx = np.unique(output['ID'], return_index=True)[1]
        for info in infos:
            output[info] = output[info][unique_idx]
        #print('After removing duplicates, the number of stars is:', len(output['ID']))
        print('Star metadata in Snapshot Idx %s is extracted' % idx)
        np.save(metadata_dir + '/star_metadata_allbox_'+str(idx)+'.npy', output)
    return None

def stars_assignment(rawtree, pfs, metadata_dir, print_mode = True):
    """
    This function uniquely assigns each star in the simulation box to a halo. 
    There are two steps:
    + Step 1: Locate the halo where a star is born in. If a star is born in the intersection of multiple halos, perform energy calculation to see which halo that star belongs to. Assume that that star remains in that halo until the end of the simulation. If that halo is a sub-halo, add that star to the main halo when the two halos merge. This step helps speed up the star assignment process because we don't need to calculate the orbital energy of each star.
    + Step 2: Re-evaluate the assumption and output from Step 1. If a star moves outside of the in-situ halo at a certain timestep (hereby called "reassign star"), remove that star from that halo, and find whether that star is bound to another halo. This steps require enegy calculation for each reassigned star, but the number of reassigned stars is much smaller than the total number of stars.
    ---
    Input
    ---
    rawtree: 
      the SHINBAD merger tree output (can be smoothed if called)
    pfs: 
      the list of the snapshot's directory
    halo_dir:
        the directory to the halo finding and the refined region output
    metadata_dir: 
      the directory to the file containing the star's metadata
    numsegs:
        the number of segments to divide the box into. This is used if the star metadata needs to be extracted
    print_mode:
        whether to print the output of the function (for debugging purpose)
    ---
    Output
    ---
    output_final: 
      a dictionary containing the halos with the ID of their stars. The keys of the dictionary are the Snapshot indices.
      Each snapshot index is another dictary whose keys are the branches with stars, the SFR, and the total stellar mass.
    """
    if glob.glob(metadata_dir + '/' + 'star_metadata_allbox_*.npy') == [] or os.path.exists(metadata_dir + '/' + 'stars_assignment_step1_backup.npy') == False or os.path.exists(metadata_dir + '/' + 'halo_wstars_map.npy') == False: 
        halo_wstars_map = {}
        output = {}
        for idx in range(0, len(pfs)):
            output[idx] = {}
        highvel_IDs = np.array([])
        starting_idx = 0
    else:
        halo_wstars_map = np.load(metadata_dir + '/' + 'halo_wstars_map.npy', allow_pickle=True).tolist()
        output = np.load(metadata_dir + '/' + 'stars_assignment_step1_backup.npy', allow_pickle=True).tolist()
        highvel_IDs = np.load(metadata_dir + '/' + 'highvel_IDs.npy', allow_pickle=True).tolist()
        starting_idx = list(halo_wstars_map.keys())[-1] + 1
    #------------------------------------------------------------------------
    for idx in range(starting_idx, len(pfs)):
        #
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        pos_all = metadata['pos']
        ID_all = metadata['ID'].astype(int)
        vel_all = metadata['vel']*1e3 #convert from km/s to m/s
        if idx == 0 or len(list(output[idx].keys())) == 0:
            ID_all_prev = np.array([])
        else:
            ID_all_prev = np.concatenate(list(output[idx].values())).astype(int) #these are the ID of the stars that are already assigned to halos in the previous snapshot. This also helps address the issue of a main progenitor branch ending before the last snapshot.
        #
        ID_unassign = np.setdiff1d(ID_all, ID_all_prev)
        pos_unassign = pos_all[np.intersect1d(ID_all, ID_unassign, return_indices=True)[1]]
        vel_unassign = vel_all[np.intersect1d(ID_all, ID_unassign, return_indices=True)[1]]
        #Obtain the halos with stars
        halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch = list_of_halos_wstars_idx(rawtree, pos_all, idx)
        halo_wstars_map[idx] = {} #stored it for later used in Step 2 of the code
        halo_wstars_map[idx]['pos'] = halo_wstars_pos
        halo_wstars_map[idx]['rvir'] = halo_wstars_rvir
        halo_wstars_map[idx]['branch_wstars'] = halo_wstars_branch
        #
        #The shape of halo_boolean is (X,Y), where X is the number of star particles and Y is the number of halos with stars
        halo_boolean = np.linalg.norm(pos_unassign[:, np.newaxis, :] - halo_wstars_pos, axis=2) <= halo_wstars_rvir
        #The number of halos a star particle is in. For example, if this value = 2, the star particle is in the region of 2 halos
        overlap_boolean = np.sum(halo_boolean, axis=1) 
        #
        ID_overlap = ID_unassign[overlap_boolean > 1]
        ID_overlap = np.append(ID_overlap, np.intersect1d(ID_unassign,highvel_IDs)) #need to re-evaluate the highvel_IDs even if they are in the region of only 1 halo.
        ID_overlap = np.unique(ID_overlap)
        halo_boolean_overlap = halo_boolean[np.intersect1d(ID_unassign, ID_overlap, return_indices=True)[1]]
        #
        ID_indp = ID_unassign[overlap_boolean == 1]
        ID_indp = np.setdiff1d(ID_indp, highvel_IDs) #need to re-evaluate the highvel_IDs even if they are in the region of only 1 halo.
        halo_boolean_indp = halo_boolean[np.intersect1d(ID_unassign, ID_indp, return_indices=True)[1]]
        #
        #The list of stars in each halo's region
        starmap_ID = []
        for j in range(halo_boolean_indp.shape[1]):
            starmap_ID.append(ID_indp[halo_boolean_indp[:,j]])
        #
        if len(ID_overlap) > 0:
            ds = yt.load(pfs[idx])
            pos_overlap = pos_unassign[np.intersect1d(ID_unassign, ID_overlap, return_indices=True)[1]]
            vel_overlap = vel_unassign[np.intersect1d(ID_unassign, ID_overlap, return_indices=True)[1]]
            #overlap_energy_map is a dictionary that contains the energy of a star in each of its overlap regions
            overlap_energy_map = collections.defaultdict(list)
            #this for loop calculate the energy for all overlapped stars that are in the same halo to speed up time
            for i_branch in range(len(halo_wstars_branch)):
                #Select the IDs that are in the same halo and the same time step for the energy calculation
                ID_for_erg = ID_overlap[halo_boolean_overlap[:,i_branch]]
                if len(ID_for_erg) > 0:
                    pos_for_erg = pos_overlap[halo_boolean_overlap[:,i_branch]]
                    vel_for_erg = vel_overlap[halo_boolean_overlap[:,i_branch]]
                    E = find_total_E(pos_for_erg, vel_for_erg, ds, rawtree, halo_wstars_branch[i_branch], idx)
                    for k in range(len(ID_for_erg)):
                        overlap_energy_map[ID_for_erg[k]].append(E[k])
            for k in range(len(ID_overlap)):
                overlap_branch = halo_wstars_branch[halo_boolean_overlap[k]]
                E_list = overlap_energy_map[ID_overlap[k]]
                if len(E_list) == 0:
                    continue
                if np.min(E_list) < 0:
                    bound_branch = overlap_branch[np.argmin(E_list)]
                    starmap_ID[list(halo_wstars_branch).index(bound_branch)] = np.append(starmap_ID[list(halo_wstars_branch).index(bound_branch)], ID_overlap[k]) 
                    print('For Star %s, the overlapped branches are %s and the energies are %s. This star is assigned to Branch %s.' % (int(ID_overlap[k]), overlap_branch, E_list, bound_branch))
                else:
                    print('For Star %s, the overlapped branches are %s and the energies are %s. This star is NOT bound to any branches.' % (int(ID_overlap[k]), overlap_branch, E_list))
                    highvel_IDs = np.append(highvel_IDs, ID_overlap[k])
                    highvel_IDs = np.unique(highvel_IDs)
        len_starmap = [len(i) for i in starmap_ID]
        # Add stars to subsequent snapshots
        for i in range(len(halo_wstars_branch)):
            if len(starmap_ID[i]) > 0: 
                for j in extract_and_order_snapshotIdx(rawtree, halo_wstars_branch[i]): #assuming when a star forms inside a halo, it will not leave that halo 
                    if int(j) >= idx:
                        if halo_wstars_branch[i] not in output[j].keys():
                            output[j][halo_wstars_branch[i]] = starmap_ID[i]
                        else:
                            output[j][halo_wstars_branch[i]] = np.append(output[j][halo_wstars_branch[i]], starmap_ID[i])
                #for subbranch (or deeper sub-branch), the stars in that sub-branch will belong to the branch at lower level after the two halos merge
                nlevels = halo_wstars_branch[i].count('_')
                if nlevels > 1:
                    print('DEEPER SUB-BRANCHES DETECTED')
                loop_branch = halo_wstars_branch[i]
                for level in range(nlevels): #add the stars in the sub-branch to higher branches
                    deepest_lvl = loop_branch.split('_')[-1]
                    mainbranch = loop_branch.split('_' + deepest_lvl)[0]
                    merge_timestep = np.max(extract_and_order_snapshotIdx(rawtree, loop_branch)) + 1
                    last_timestep = np.max(extract_and_order_snapshotIdx(rawtree, mainbranch))
                    for j in range(merge_timestep, last_timestep + 1):
                        if mainbranch not in output[j].keys():
                            output[j][mainbranch] = starmap_ID[i]
                        else:
                            output[j][mainbranch] = np.append(output[j][mainbranch], starmap_ID[i])
                    loop_branch = mainbranch
        #
        #ID_all_prev = ID_all
        np.save('%s/stars_assignment_step1_backup.npy' % (metadata_dir), output)
        np.save('%s/halo_wstars_map.npy' % (metadata_dir), halo_wstars_map)
        np.save('%s/highvel_IDs.npy' % (metadata_dir), highvel_IDs)
        #
        if print_mode == True:
            print(idx, 'Number of total unassigned stars is:', len(ID_unassign))
            print('Number of overlapped stars is', len(ID_overlap), ', Number of independent stars is', len(ID_indp))
            print('Halo with stars:', halo_wstars_branch)
            #For brevity, only print the halos with assigned stars in them at this timestep
            print('Number of assingned stars in each halo:', dict(zip(np.array(halo_wstars_branch)[np.array(len_starmap) != 0], np.array(len_starmap)[np.array(len_starmap) != 0])), '\n') 
        #Free some memory
        del metadata, pos_all, ID_all, vel_all, ID_unassign, pos_unassign, vel_unassign, halo_boolean, overlap_boolean, ID_indp, starmap_ID
        if len(ID_overlap) > 0:
            del ID_overlap, ds, pos_overlap, vel_overlap, overlap_energy_map, ID_for_erg, pos_for_erg, vel_for_erg, E, E_list
    #------------------------------------------------------------------------
    #This step removes the stars that moves outside of the halo's virial radius and addes them to another halos if needed. 
    #The unique stellar mass and SFR is also calculated in this step. 
    #print(halo_wstars_map)
    #print(output)
    if os.path.exists(metadata_dir + '/' + 'stars_assignment_step2_backup.npy') == False:
        output_final = {} #the re-analyzed output
        reassign_dict = {} #the list of stars that need to be re-assigned during step 2 of the code
        for idx in range(0, len(pfs)):
            reassign_dict[idx] = np.array([]).astype(int)
        prev_halo_map = collections.defaultdict(list) #the dictionary containing the branch each star belongs to in Step 1 of the code
        starting_idx_step2 = 0
    else:
        output_final = np.load(metadata_dir + '/' + 'stars_assignment_step2_backup.npy', allow_pickle=True).tolist()
        reassign_dict = np.load(metadata_dir + '/' + 'reassign_dict_step2.npy', allow_pickle=True).tolist()
        prev_halo_map = np.load(metadata_dir + '/' + 'prev_halo_map_step2.npy', allow_pickle=True).tolist()
        starting_idx_step2 = list(output_final.keys())[-1] + 1
    for idx in range(starting_idx_step2, len(pfs)):
        output_final[idx] = {}
        ds = yt.load(pfs[idx])
        #
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        pos_all = metadata['pos']
        ID_all = metadata['ID'].astype(int)
        vel_all = metadata['vel']*1e3 #convert from km/s to m/s
        for branch in output[idx].keys():
            if idx not in extract_and_order_snapshotIdx(rawtree, branch):
                continue
            ID = output[idx][branch]
            #obtain the stars found in the initial output
            pos = pos_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            ID = ID_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            vel = vel_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            #
            halo_center = rawtree[branch][idx]['Halo_Center']
            halo_radius = get_r200_radius(rawtree, branch, idx)
            #
            #remain_bool: stars that still remain in the halo where they are born
            #loss_bool: stars that move out of the halo where they were born 
            remain_bool = np.linalg.norm(pos - halo_center, axis=1) < halo_radius
            loss_bool = np.linalg.norm(pos - halo_center, axis=1) >= halo_radius
            #------------------------
            #Reassign the "loss" stars to new halos by using bound energy condition. Note that when a star is lose, we will check its energy for the rest of the timestep. 
            ID_loss = ID[loss_bool]
            for ID_loss_i in ID_loss:
                if ID_loss_i not in prev_halo_map.keys():
                    prev_halo_map[ID_loss_i] = branch
            if len(ID_loss) > 0:
                for j in range(idx, max(extract_and_order_snapshotIdx(rawtree, branch)) + 1):
                    reassign_dict[j] = np.append(reassign_dict[j], ID_loss)
                    reassign_dict[j] = np.unique(reassign_dict[j])
            #------------------------
            ID_remain = ID[remain_bool]
            ID_remain = np.setdiff1d(ID_remain, reassign_dict[idx])
            output_final[idx][branch] = {}
            output_final[idx][branch]['ID'] = ID_remain
        #-------------------------
        #reassign_energy_map is a dictionary that contains the energy of a star gets outside of its first assigned halo and move to another halo region
        #The logic here is similar to how we calculate the energy for the overlapped stars
        reassign_energy_map = collections.defaultdict(list)
        pos_reassign = pos_all[np.intersect1d(ID_all, reassign_dict[idx], return_indices=True)[1]]
        vel_reassign = vel_all[np.intersect1d(ID_all, reassign_dict[idx], return_indices=True)[1]]
        ID_reassign = ID_all[np.intersect1d(ID_all, reassign_dict[idx], return_indices=True)[1]]
        print('At Snapshot', idx, ', %s stars need to be re-assigned' % len(reassign_dict[idx]))
        del pos_all, vel_all
        halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch = halo_wstars_map[idx].values() #obtain the list of halos with stars, the halo_wstars_map is computed above
        halo_boolean_reassign = np.linalg.norm(pos_reassign[:, np.newaxis, :] - halo_wstars_pos, axis=2) <= halo_wstars_rvir
        for i_branch in range(len(halo_wstars_branch)):
            ID_for_erg = ID_reassign[halo_boolean_reassign[:,i_branch]]
            if len(ID_for_erg) > 0:
                pos_for_erg = pos_reassign[halo_boolean_reassign[:,i_branch]]
                vel_for_erg = vel_reassign[halo_boolean_reassign[:,i_branch]]
                E = find_total_E(pos_for_erg, vel_for_erg, ds, rawtree, halo_wstars_branch[i_branch], idx)
                for k in range(len(ID_for_erg)):
                    reassign_energy_map[ID_for_erg[k]].append(E[k])
        for k in range(len(ID_reassign)):
            reassign_branch = halo_wstars_branch[halo_boolean_reassign[k]] #these are the branches that the reassigned stars move to (before the reassignment and energy calculation)
            if ID_reassign[k] in reassign_energy_map.keys():
                E_list = reassign_energy_map[ID_reassign[k]]
                if np.min(E_list) < 0:
                    new_bound_branch = reassign_branch[np.argmin(E_list)]
                    print('At Snapshot', idx, 'Star', ID_reassign[k], 'move from Branch', prev_halo_map[ID_reassign[k]], 'to', new_bound_branch)
                    if new_bound_branch not in output_final[idx].keys(): #add the stars bounded with the new halo to the output_final
                        output_final[idx][new_bound_branch] = {}
                        output_final[idx][new_bound_branch]['ID'] = np.array([ID_reassign[k]])
                    else:
                        output_final[idx][new_bound_branch]['ID'] = np.append(output_final[idx][new_bound_branch]['ID'], ID_reassign[k])
            else:
                continue #the star is not bound to any halo, skip this star  
        #Save for backup
        np.save('%s/stars_assignment_step2_backup.npy' % (metadata_dir), output_final)
        np.save('%s/reassign_dict_step2.npy' % (metadata_dir), reassign_dict)
        np.save('%s/prev_halo_map_step2.npy' % (metadata_dir), prev_halo_map)
    #Finalize the output_final star ID and calculate the unique total stellar mass and SFR.
    for idx in output_final.keys():
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        mass_all = metadata['mass']
        age_all = metadata['age']
        ID_all = metadata['ID']
        type_all = metadata['type']
        pos_all = metadata['pos']
        for branch in output_final[idx].keys():
            ID = output_final[idx][branch]['ID']
            mass = mass_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            age = age_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            type = type_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            pos = pos_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            #
            mass2 = mass[type == 7]
            age2 = age[type == 7]
            positions2 = pos[type == 7]
            id2 = ID[type == 7]
            mass3 = mass[type == 5]
            age3 = age[type == 5]
            positions3 = pos[type == 5]
            id3 = ID[type == 5]
            #
            output_final[idx][branch]['total_mass'] = np.sum(mass)
            output_final[idx][branch]['sfr'] = np.sum(mass[age < 0.01])/1e7
            output_final[idx][branch]['mass2'] = mass2
            output_final[idx][branch]['age2'] = age2
            output_final[idx][branch]['positions2'] = positions2
            output_final[idx][branch]['id2'] = id2
            output_final[idx][branch]['mass3'] = mass3
            output_final[idx][branch]['age3'] = age3
            output_final[idx][branch]['positions3'] = positions3
            output_final[idx][branch]['id3'] = id3
    return output_final

def branch_first_rearrange(output_final):
    output_re = {}
    for snapshot in output_final.keys():
        for branch in output_final[snapshot].keys():
            if branch not in output_re.keys():
                output_re[branch] = {}
                output_re[branch][snapshot] = output_final[snapshot][branch]
            else:
                output_re[branch][snapshot] = output_final[snapshot][branch]
    return output_re

            
if __name__ == "__main__":
    if nprocs < 128:
        numsegs = 10
    else:
        numsegs = int(np.ceil((nprocs*5)**(1/3)) + 1)
    halo_dir = sys.argv[1]
    metadata_dir = sys.argv[2]
    halotree_ver = sys.argv[3]
    #
    rawtree = np.load(halo_dir + '/halotree_%s_final.npy' % halotree_ver, allow_pickle=True).tolist()
    pfs = np.loadtxt(halo_dir + '/pfs_allsnaps_%s.txt' % halotree_ver, dtype=str)[:,0]
    if yt.is_root():
        print('Done loading data')
        print(metadata_dir)
    #
    #This is to extract the star metadata from the simulation box
    for idx in range(0, len(pfs)):
        if os.path.exists(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx) == False:
            if yt.is_root():
                print('Starting to extract metadata from Snapshot %s' % idx)
            extract_star_metadata(pfs, idx, numsegs, halo_dir, metadata_dir)
    #
    if yt.is_root():
        #
        stars_assign_output = stars_assignment(rawtree, pfs, metadata_dir, print_mode = True)
        np.save(metadata_dir + '/stars_assignment_snapFirst.npy', stars_assign_output)
        #
        #This is to re-arange the data structure to match with Kirk's pipeline
        branch_first = True
        if branch_first == True:
            stars_assign_output_re = branch_first_rearrange(stars_assign_output)
            np.save(metadata_dir + '/stars_assignment_branchFirst.npy', stars_assign_output_re)