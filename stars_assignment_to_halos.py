import numpy as np
import yt
from astropy.constants import G
import astropy.units as u
from tqdm import tqdm
import os

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
    oden_list = np.array([100, 150, 200, 250, 300, 500, 700])
    char_radius_list = np.array([])
    for oden in oden_list:
        key = 'r%s' % oden
        char_radius_list = np.append(char_radius_list, rawtree[branch][idx][key])
    return oden_list, char_radius_list

def find_total_E(star_pos, star_vel, ds, rawtree, branch, idx):
    #this function calculate the total orbital energy of a star around a halo
    #the unit of position is code_length and the unit of velocity is code_length/s
    star_r_codelength = np.linalg.norm(star_pos - rawtree[branch][idx]['Halo_Center'])
    star_r = (star_r_codelength*ds.units.code_length).to('m').v
    #
    halo_vel = (rawtree[branch][idx]['Vel_Com']*ds.units.code_length/ds.units.s).to('m/s').v
    star_relvel_mag = np.linalg.norm(star_vel - halo_vel)
    #Kinetic energy
    KE = 0.5*star_relvel_mag**2
    #Approximate M(r < star_r) by using the overdensity
    oden_list, char_radius_list = extract_char_radius(rawtree, branch, idx)
    char_radius_list = (char_radius_list*ds.units.code_length).to('m').v
    oden = oden_list[char_radius_list > star_r][-1]
    M = (4/3)*np.pi*oden*univDen(ds)*star_r**3
    PE = -G.value*M/star_r
    E = KE + PE
    return E

def extract_star_metadata(pfs, idx):
    ds = yt.load(pfs[idx])
    ll_all = ds.domain_left_edge.to('code_length').v
    ur_all = ds.domain_right_edge.to('code_length').v
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
        reg = ds.box(ll[i] ,ur[i])
        ptype = reg['all', 'particle_type'].v
        pmass = reg['all', 'particle_mass'].to('Msun').v
        ppos = reg['all', 'particle_position'].to('code_length').v
        pvel = reg['all', 'particle_velocity'].to('km/s').v
        pftime = reg['all','creation_time'].to('Gyr').v
        page = reg['all','age'].to('Gyr').v
        pmet = reg['all','metallicity_fraction'].to('Zsun').v
        pID = reg['all','particle_index'].v
        #
        star_bool = np.logical_and(np.logical_or(ptype == 5, ptype == 7), pmass > 1)
        star_mass = pmass[star_bool]
        star_pos = ppos[star_bool]
        star_vel = pvel[star_bool]
        star_ftime = pftime[star_bool]
        star_age = page[star_bool]
        star_met = pmet[star_bool]
        star_ID = pID[star_bool]
        #Export the output
        output = {}
        output['mass'] = star_mass
        output['pos'] = star_pos
        output['vel'] = star_vel
        output['ftime'] = star_ftime
        output['age'] = star_age
        output['met'] = star_met
        output['ID'] = star_ID
        print('Snapshot Idx %s is finished' % idx)
        np.save('/work/hdd/bdax/gtg115x/new_zoom_5/box_2_z_1_no-shield_temp/star_metadata/star_metadata_allbox_'+str(idx)+'.npy', output)
        del ds, reg, ptype, pmass, ppos, page, pmet, pID, pvel, pftime,  star_bool, star_mass, star_pos, star_age, star_met, star_ID, star_vel, star_ftime, output


def stars_assignment(rawtree, pfs, metadata_dir, print_mode = True):
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
    metadata_dir: 
      the directory to the file containing the star's metadata
    ---
    Output
    ---
    output_final: 
      a dictionary containing the halos with the ID of their stars. The keys of the dictionary are the Snapshot indices.
      Each snapshot index is another dictary whose keys are the branches with stars, the SFR, and the total stellar mass.
    """
    halo_wstars_map = {}
    output = {}
    for idx in range(0, len(pfs)):
        output[idx] = {}

    #------------------------------------------------------------------------
    for idx in tqdm(range(0, len(pfs))):
        #
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        pos_all = metadata['pos']
        age_all = metadata['age']
        mass_all = metadata['mass']
        ID_all = metadata['ID']
        vel_all = metadata['vel']*1e3 #convert from km/s to m/s
        #ID_all = np.array(np.load(metadata_dir + '/'  + 'star_ID_allbox_%s.npy' % idx, allow_pickle=True).tolist()).astype(int)
        #if os.path.exists(metadata_dir + '/' + 'star_vel_allbox_%s.npy' % idx) == True:
        #    vel_all = np.array(np.load(metadata_dir + '/' + 'star_vel_allbox_%s.npy' % idx, allow_pickle=True).tolist()['vel'])
        #else:
        #    vel_all = np.empty(shape=(0,3))
        #
        if idx == 0:
            ID_all_prev = np.array([])
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
                bound_branch = overlap_branch[np.argmin(E_list)] #in this step, we don't check whether the total energy of each star is negative, so we also don't check it here. 
                starmap_ID[list(halo_wstars_branch).index(bound_branch)] = np.append(starmap_ID[list(halo_wstars_branch).index(bound_branch)], ID_overlap[k])
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
        ID_all_prev = ID_all
        #
        if print_mode == True:
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
        metadata = np.load(metadata_dir + '/' + 'star_metadata_allbox_%s.npy' % idx, allow_pickle=True).tolist()
        pos_all = metadata['pos']
        mass_all = metadata['mass']
        age_all = metadata['age']
        ID_all = metadata['ID']
        vel_all = metadata['vel']*1e3 #convert from km/s to m/s
        #ID_all = np.array(np.load(metadata_dir + '/' + 'star_ID_allbox_%s.npy' % idx, allow_pickle=True).tolist()).astype(int)
        #if os.path.exists(metadata_dir + '/' + 'star_vel_allbox_%s.npy' % idx) == True:
        #    vel_all = np.array(np.load(metadata_dir + '/' + 'star_vel_allbox_%s.npy' % idx, allow_pickle=True).tolist()['vel'])
        #else:
        #    vel_all = np.empty(shape=(0,3))
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
            #---------------------------
            #Reassign the "loss" stars to new halos by using bound energy condition
            ID_loss = ID[loss_bool]
            pos_loss = pos[loss_bool]
            vel_loss = vel[loss_bool]
            if len(ID_loss) > 0:
                halo_wstars_pos, halo_wstars_rvir, halo_wstars_branch = halo_wstars_map[idx] #obtain the list of halos with stars, the halo_wstars_map is computed above
                halo_boolean = np.linalg.norm(pos_loss[:, np.newaxis, :] - halo_wstars_pos, axis=2) <= halo_wstars_rvir
                ds = yt.load(pfs[idx])
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
        #ID_all = np.array(np.load(metadata_dir + '/' + 'star_ID_allbox_%s.npy' % idx, allow_pickle=True).tolist()).astype(int)
        for branch in output_final[idx].keys():
            ID = output_final[idx][branch]['ID']
            mass = mass_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            age = age_all[np.intersect1d(ID_all, ID, return_indices=True)[1]]
            output_final[idx][branch]['total_mass'] = np.sum(mass)
            output_final[idx][branch]['sfr'] = np.sum(mass[age < 0.01])/1e7
    return output_final
            
if __name__ == "__main__":
    rawtree = np.load('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_no-shield_temp/halotree_1088_final.npy', allow_pickle=True).tolist()
    pfs = np.loadtxt('/work/hdd/bdax/gtg115x/Halo_Finding/box_2_z_1_no-shield_temp/pfs_allsnaps_1088.txt', dtype=str)[:,0]
    metadata_dir = '/work/hdd/bdax/gtg115x/new_zoom_5/box_2_z_1_no-shield_temp/star_metadata'
    stars_assign_output = stars_assignment(rawtree, pfs, metadata_dir, print_mode = True)
    np.save('/work/hdd/bdax/gtg115x/new_zoom_5/box_2_z_1_no-shield_temp/stars_assignment.npy', stars_assign_output)
