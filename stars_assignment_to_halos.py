import numpy as np
import yt
from astropy.constants import G
import astropy.units as u
from tqdm import tqdm
import os
import glob as glob
from scipy.interpolate import CubicSpline
import healpy as hp
from scipy.spatial.distance import cdist
import time as time_sys

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

def list_of_halos_wstars_idx(rawtree, pos_allstars, idx):
    halo_wstars_pos = np.empty(shape=(0,3))
    halo_wstars_rvir = np.array([])
    halo_wstars_branch = np.array([])
    for branch, vals in rawtree.items():
        if idx in vals.keys():
            if (np.linalg.norm(pos_allstars - rawtree[branch][idx]['Halo_Center'], axis=1) < rawtree[branch][idx]['Halo_Radius']).any():
                halo_wstars_pos = np.vstack((halo_wstars_pos, vals[idx]['Halo_Center']))
                halo_wstars_rvir = np.append(halo_wstars_rvir, vals[idx]['Halo_Radius'])
                halo_wstars_branch = np.append(halo_wstars_branch, branch)   
    return halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch

def univDen(ds):
    # Hubble constant
    H0 = ds.hubble_constant * 100 * u.km/u.s/u.Mpc
    H = H0**2 * (ds.omega_matter*(1 + ds.current_redshift)**3 + ds.omega_lambda)  # Technically H^2
    G = 6.67e-11 * u.m**3/u.s**2/u.kg
    # Density of the universe
    den = (3*H/(8*np.pi*G)).to("kg/m**3") / u.kg * u.m**3
    return den.value

def extract_char_radius(rawtree, branch, idx):
    oden_list = np.array([150, 200, 250, 300, 500, 700, 1000])
    char_radius_list = np.array([])
    for oden in oden_list:
        key = 'r%s' % oden
        char_radius_list = np.append(char_radius_list, rawtree[branch][idx][key])
    return oden_list, char_radius_list

def vecs_calc(nside):
    pix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside,np.arange(pix))
    vecs = hp.ang2vec(np.array(theta),np.array(phi))
    return vecs

def cut_particles(pos,mass,center,ids,idl_i=None,cut_size=700,dense=False,segments=1,timing=0):
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
    if idl_i is not None:
            bool = np.logical_or(np.isin(ids,idl_i),bool)
    return mass[bool], bool

def find_total_E(star_pos, star_vel, ds, rawtree, branch, idx):
    #
    #This function finds the total energy of an array of star particles in one halo at a certain timestep.
    #
    regA = ds.sphere(rawtree[branch][idx]['Halo_Center'], rawtree[branch][idx]['Halo_Radius'])
    #
    massA = regA['all','particle_mass'].to('kg')
    posA = regA['all','particle_position'].to('m')
    velA = regA['all','particle_velocity'].to('m/s')
    idsA = regA['all','particle_index'].v
    posA = posA[massA.to('Msun') > 1]
    velA = velA[massA.to('Msun') > 1]
    idsA = idsA[massA.to('Msun') > 1]
    massA = massA[massA.to('Msun') > 1]
    #
    centerA = (rawtree[branch][idx]['Halo_Center']*ds.units.code_length).to('m')
    #
    massA_cut, boolA_cut = cut_particles(posA.v,massA.v,centerA.v,idsA)
    posA_cut = posA[boolA_cut]
    #
    disAinv_cut = 1/np.linalg.norm((star_pos*ds.units.code_length).to('m').v[:, np.newaxis] - posA_cut.v, axis=2)
    disAinv_cut[~np.isfinite(disAinv_cut)] = 0
    #
    #disAinv_cut = 1/np.linalg.norm((star_pos*ds.units.code_length).to('m').v - posA_cut.v, axis=1)
    #disAinv_cut[~np.isfinite(disAinv_cut)] = 0
    #
    PE = np.sum(-G.value*massA_cut*disAinv_cut, axis=1)
    velcom = (rawtree[branch][idx]['Vel_Com']*ds.units.code_length/ds.units.s).to('m/s').v
    KE = 0.5*np.linalg.norm(star_vel - velcom, axis=1)**2
    E = KE + PE
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
    #load the refined region, we assume that stars only exist in this region
    region_idx = region_number(idx, halo_dir)
    refined_region = np.load(halo_dir + '/' + 'refined_region_%s.npy' % region_idx, allow_pickle=True).tolist()
    #
    ds = yt.load(pfs[idx])
    ll_all = np.array(refined_region[0])
    ur_all = np.array(refined_region[1])
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
    return output

def stars_assignment(rawtree, pfs, halo_dir, metadata_dir, numsegs, print_mode = True):
    """
    This function uniquely assigns each star in the simulation box to a halo. 
    There are two steps:
    + Step 1: Locate the halo where a star is born in. If a star is born in the intersection of multiple halos, perform energy calculation to see which halo that star belongs to. Assume that that star remains in that halo until the end of the simulation. If that halo is a sub-halo, add that star to the main halo when the two halos merge. This step helps speed up the star assignment process because we don't need to calculate the orbital energy of each star.
    + Step 2: Re-evaluate the assumption and output from Step 1. If a star moves outside of the in-situ halo at a certain timestep (hereby called "loss star"), remove that star from that halo, and find whether that star is bound to another halo. This steps require enegy calculation for each loss star, but the number of loss stars is much smaller than the total number of stars.
    ---
    Input
    ---
    rawtree: 
      the SHINBAD merger tree output
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
    if glob.glob(metadata_dir + '/' + 'star_metadata_allbox_*.npy') == [] or os.path.exists(metadata_dir + '/' + 'stars_assignment_backup.npy') == False or os.path.exists(metadata_dir + '/' + 'halo_wstars_map.npy') == False: 
        halo_wstars_map = {}
        output = {}
        for idx in range(0, len(pfs)):
            output[idx] = {}
        starting_idx = 0
        restart_flag = False
    else:
        halo_wstars_map = np.load(metadata_dir + '/' + 'halo_wstars_map.npy', allow_pickle=True).tolist()
        output = np.load(metadata_dir + '/' + 'stars_assignment_backup.npy', allow_pickle=True).tolist()
        starting_idx = list(halo_wstars_map.keys())[-1] + 1
        restart_flag = True
    time_sys.sleep(100)
    #------------------------------------------------------------------------
    for idx in tqdm(range(starting_idx, len(pfs))):
        #
        if os.path.exists(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx) == False:
            if yt.is_root():
                print('Starting to load the snapshot and extract metadata from Snapshot %s' % idx)
            metadata = extract_star_metadata(pfs, idx, numsegs, halo_dir, metadata_dir)
        else:
            metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        pos_all = metadata['pos']
        age_all = metadata['age']
        mass_all = metadata['mass']
        ID_all = metadata['ID'].astype(int)
        vel_all = metadata['vel']*1e3 #convert from km/s to m/s
        if idx == 0:
            ID_all_prev = np.array([])
        elif restart_flag == True:
            ID_all_prev = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % (idx - 1), allow_pickle=True).tolist()['ID'].astype(int)
        #
        ID_unassign = np.setdiff1d(ID_all, ID_all_prev)
        pos_unassign = pos_all[np.intersect1d(ID_all, ID_unassign, return_indices=True)[1]]
        vel_unassign = vel_all[np.intersect1d(ID_all, ID_unassign, return_indices=True)[1]]
        #Obtain the halos with stars
        halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch = list_of_halos_wstars_idx(rawtree, pos_all, idx)
        halo_wstars_map[idx] = (halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch) #stored it for later used in Step 2 of the code
        #
        #The shape of halo_boolean is (X,Y), where X is the number of star particles and Y is the number of halos with stars
        halo_boolean = np.linalg.norm(pos_unassign[:, np.newaxis, :] - halo_wstars_pos, axis=2) <= halo_wstars_rvir
        #The number of halos a star particle is in. For example, if this value = 2, the star particle is in the region of 2 halos
        overlap_boolean = np.sum(halo_boolean, axis=1) 
        #
        ID_overlap = ID_unassign[overlap_boolean > 1]
        halo_boolean_overlap = halo_boolean[overlap_boolean > 1]
        ID_indp = ID_unassign[overlap_boolean == 1]
        halo_boolean_indp = halo_boolean[overlap_boolean == 1]
        #
        #The list of stars in each halo's region
        starmap_ID = []
        for j in range(halo_boolean_indp.shape[1]):
            starmap_ID.append(ID_indp[halo_boolean_indp[:,j]])
        #
        if len(ID_overlap) > 0:
            ds = yt.load(pfs[idx])
            pos_overlap = pos_unassign[overlap_boolean > 1]
            vel_overlap = vel_unassign[overlap_boolean > 1]
            overlap_branch_total = []
            for k in range(len(ID_overlap)):
                overlap_branch = halo_wstars_branch[halo_boolean_overlap[k]]
                E_list = np.array([])
                for branch in overlap_branch:
                    overlap_branch_total.append(branch)
                    E = find_total_E(pos_overlap[k], vel_overlap[k], ds, rawtree, branch, idx)
                    E_list = np.append(E_list, E)
                if yt.is_root():
                    print('For this overlapped Star %s, the overlapped branches are %s and the corresponding energies are %s' % (ID_overlap[k], overlap_branch, E_list))
                bound_branch = overlap_branch[np.argmin(E_list)] #in this step, we don't check whether the total energy of each star is negative, so we also don't check it here. 
                starmap_ID[list(halo_wstars_branch).index(bound_branch)] = np.append(starmap_ID[list(halo_wstars_branch).index(bound_branch)], ID_overlap[k])
            if yt.is_root():
                print('OVERLAP DETECTED AT BRANCHES', set(overlap_branch_total))
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
                if nlevels > 1 and yt.is_root():
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
        ID_all_prev = ID_all
        if yt.is_root():
            np.save('%s/stars_assignment_backup.npy' % (metadata_dir), output)
            np.save('%s/halo_wstars_map.npy' % (metadata_dir), halo_wstars_map)
        #
        if print_mode == True and yt.is_root():
            print(idx, 'Number of total unassigned stars is:', len(ID_unassign))
            print('Number of overlapped stars is', len(ID_overlap), ', Number of independent stars is', len(ID_indp))
            print('Halo with stars:', halo_wstars_branch)
            print('Number of assingned stars in each halo:', len_starmap, '\n')
    #------------------------------------------------------------------------
    #This step removes the stars that moves outside of the halo's virial radius and addes them to another halos if needed. 
    #The unique stellar mass and SFR is also calculated in this step. 
    #print(halo_wstars_map)
    #print(output)
    output_final = {} #the re-analyzed output
    for idx in output.keys():
        output_final[idx] = {}
        ds = yt.load(pfs[idx])
        length_unit_pc = ds.domain_right_edge[0].to('pc').v.tolist()
        #
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        pos_all = metadata['pos']
        mass_all = metadata['mass']
        age_all = metadata['age']
        ID_all = metadata['ID']
        vel_all = metadata['vel']*1e3 #convert from km/s to m/s
        for branch in output[idx].keys():
            ID = output[idx][branch]
            #obtain the stars found in the initial output
            pos = pos_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            mass = mass_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            age = age_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            ID = ID_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            vel = vel_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            #
            halo_center = rawtree[branch][idx]['Halo_Center']
            halo_radius = rawtree[branch][idx]['Halo_Radius']
            #
            #remain_bool: stars that still remain in the halo where they are born
            #loss_bool: stars that move out of the halo where they were born 
            remain_bool = np.linalg.norm(pos - halo_center, axis=1) < halo_radius
            loss_bool = np.linalg.norm(pos - halo_center, axis=1) >= halo_radius
            #------------------------
            ID_remain = ID[remain_bool]
            output_final[idx][branch] = {}
            output_final[idx][branch]['ID'] = ID_remain
            output_final[idx][branch]['length_unit_pc'] = length_unit_pc
            #---------------------------
            #Reassign the "loss" stars to new halos by using bound energy condition
            ID_loss = ID[loss_bool]
            pos_loss = pos[loss_bool]
            vel_loss = vel[loss_bool]
            if len(ID_loss) > 0:
                if yt.is_root():
                    print('At Snapshot', idx, 'and Branch', branch, ', %s stars move out of the halo' % len(ID_loss))
                halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch = halo_wstars_map[idx] #obtain the list of halos with stars, the halo_wstars_map is computed above
                halo_boolean = np.linalg.norm(pos_loss[:, np.newaxis, :] - halo_wstars_pos, axis=2) <= halo_wstars_rvir
                inside_branch_total = []
                #loop through each loss star
                for k in range(len(ID_loss)): 
                    inside_branch = halo_wstars_branch[halo_boolean[k]] #these are the branches that the loss stars move to
                    E_list = np.array([])
                    for ibranch in inside_branch: #perform energy calculation to see which new halo those loss stars are bound to
                        inside_branch_total.append(ibranch)
                        E = find_total_E(pos_loss[k], vel_loss[k], ds, rawtree, ibranch, idx)
                        E_list = np.append(E_list, E)
                    E_list = np.array(E_list)
                    if (E_list < 0).any() == True: #if the star is bound to multiple halos, select the one with the most negative total energy
                        bound_branch = inside_branch[np.argmin(E_list)]
                        if yt.is_root():
                            print('At Snapshot', idx, 'Star', ID_loss[k], 'move from Branch', branch, 'to', bound_branch)
                        if bound_branch not in output_final[idx].keys(): #add the stars bounded with the new halo to the output_final
                            output_final[idx][bound_branch] = {}
                            output_final[idx][bound_branch]['ID'] = np.array([ID_loss[k]])
                        else:
                            output_final[idx][bound_branch]['ID'] = np.append(output_final[idx][bound_branch]['ID'], ID_loss[k])
                    else:
                        continue #the star is not bound to any halo, skip this star            
    #Finalize the output_final star ID and calculate the unique total stellar mass and SFR.
    for idx in output_final.keys():
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        mass_all = metadata['mass']
        age_all = metadata['age']
        ID_all = metadata['ID']
        type_all = metadata['type']
        #mets_all = metadata['met']
        pos_all = metadata['pos']
        for branch in output_final[idx].keys():
            ID = output_final[idx][branch]['ID']
            mass = mass_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            age = age_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            type = type_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            #mets = mets_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            pos = pos_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            #
            mass2 = mass[type == 7]
            age2 = age[type == 7]
            #mets2 = mets[type == 7]
            positions2 = pos[type == 7]
            id2 = ID[type == 7]
            mass3 = mass[type == 5]
            age3 = age[type == 5]
            #mets3 = mets[type == 5]
            positions3 = pos[type == 5]
            id3 = ID[type == 5]
            #
            output_final[idx][branch]['total_mass'] = np.sum(mass)
            output_final[idx][branch]['sfr'] = np.sum(mass[age < 0.01])/1e7
            output_final[idx][branch]['mass2'] = mass2
            output_final[idx][branch]['age2'] = age2
            #output_final[idx][branch]['mets2'] = mets2
            output_final[idx][branch]['positions2'] = positions2
            output_final[idx][branch]['id2'] = id2
            output_final[idx][branch]['mass3'] = mass3
            output_final[idx][branch]['age3'] = age3
            #output_final[idx][branch]['mets3'] = mets3
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
    halo_dir = '/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_no-shield_temp'
    metadata_dir = '/work/hdd/bdax/tnguyen2/sandbox/stars_assignment_code_test/box_2_z_1_no-shield_temp_CubicSplineOden'
    rawtree = np.load(halo_dir + '/halotree_1405_final.npy', allow_pickle=True).tolist()
    pfs = np.loadtxt(halo_dir + '/pfs_allsnaps_1405.txt', dtype=str)[:,0]
    if yt.is_root():
        print('Done loading data')
    stars_assign_output = stars_assignment(rawtree, pfs, halo_dir, metadata_dir, numsegs, print_mode = True)
    np.save(metadata_dir + '/stars_assignment_snapFirst.npy', stars_assign_output)
    #
    #This is to re-arange the data structure to match with Kirk's pipeline
    branch_first = True
    if branch_first == True:
        stars_assign_output_re = branch_first_rearrange(stars_assign_output)
        np.save(metadata_dir + '/stars_assignment_branchFirst.npy', stars_assign_output_re)
    #Delete the temporary star_metadata_allbox files
    #rm_files = glob.glob(metadata_dir + '/' + 'star_metadata_allbox_*.npy')
    #for file in rm_files:
    #    os.remove(file)