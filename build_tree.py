import numpy as np  
import yt
import ytree
import sys,os
import glob as glob
import time

yt.enable_parallelism()
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank
nprocs = comm.size

def make_pfs(folder, rockfolder):
    """
    This function creates the pfs.dat file by matching the redshifts in the rockstar
    outputs to the redshifts in the simulation snapshots. This is used when we don't
    know what snapshot each rockstar output refers to because the pfs.dat file is
    missing. This is especially useful if the number of snapshots and the number
    of rockstar outputs are not the same.


    Parameters
    ----------
    folder : str
        The directory to the folder containing the simulation snapshots.
    rockfolder : str
        The name of the folder containing rockstar outputs.

    Returns
    -------
    None.

    """
    if yt.is_root():
        cwd = os.getcwd()
        
        #Obtaining the redshifts of all the rockstar outputs from the scales.txt file.
        #This file is produced when consistent-tree is run. Loading the redshifts this way 
        #ensures the pfs.dat snapshots matches with the tree_0_0_0.dat snapshots (because 
        #sometimes, consistent-tree removes rockstar outputs when there aren't enough halos
        #in that snapshot)
        rockstar_files = '%s/%s/outputs/scales.txt' % (folder, rockfolder)
        
        #Create a dictionary to store the redshifts of all the rockstar outputs and the output's directory
        rockstar_redshifts = []

        #Loop through each output. The scale factor is the number after the '#a =' string in the
        #output text file
        with open(rockstar_files, 'r') as file:
            for line in file:
                scale = float(line.split()[-1])
                redshift = 1/scale - 1
                rockstar_redshifts.append(redshift)
        
        #Change the directory to where the snapshots are located
        os.chdir('%s' % folder)
        
        #Obtaining the directory to all the snapshot configuration files
        bases = [["DD", "output_"],
                 ["DD", "data"],
                 ["DD", "DD"],
                 ["RD", "RedshiftOutput"],
                 ["RD", "RD"],
                 ["RS", "restart"]]
        snapshot_files = []
        for b in bases:
            snapshot_files += glob.glob("%s????/%s????" % (b[0], b[1]))
                

        #Create a dictionary to store the redshifts of all the snapshots and the snapshot's directory
        snapshot_redshifts = {}

        #Loop through each output. The scale factor is the number after the '#a =' string in the
        #output text file
        for file_dir in snapshot_files:
            with open(file_dir, 'r') as file:
                for line in file:
                    if line.startswith('CosmologyCurrentRedshift'):
                        redshift = float(line.split()[-1])
                        break
            snapshot_redshifts[file_dir] = redshift
            

        #Match the rockstar redshift to closest value in the snapshot redshift to write a pfs.dat file
        pfs_output = []
        for vals in rockstar_redshifts:
            index = np.argmin(abs(vals - np.array(list(snapshot_redshifts.values()))))
            #Obtain the directory to the snapshot whose redshift is the closest match
            snapshot_dir = list(snapshot_redshifts.keys())[index]
            pfs_output.append(snapshot_dir)

        #Write out a pfs.dat file
        with open('pfs_manual.dat','w') as file:
            for item in pfs_output:
                file.write("%s\n" % item)

        #Change back to the original working directory
        os.chdir(cwd)


def makehlist(folder,rockfolder,numtrees,depth=5,min_r=1e-3,min_mass=1e6):
    #Get the current working directory (where the output will be located) to easily switch back later
    cdw = os.getcwd()
    #Changing the directory to the consistent-tree output folder
    os.chdir('%s/%s/trees' % (folder,rockfolder))
    arbor_file = '%s/%s/trees/arbor/arbor.h5' % (folder,rockfolder)
    if os.path.exists(arbor_file):
        tree = ytree.load(arbor_file)
    #Creating .h5 format (if not yet created) to increase the performance of reading consistent-tree outputs
    else:
        tree = ytree.load('%s/%s/trees/tree_0_0_0.dat' % (folder,rockfolder))
    if yt.is_root():
        fn = tree.save_arbor()
        tree = ytree.load(arbor_file)
    #Changing back to the current working directory
    os.chdir(cdw)
      
    #Loading a snapshot parameter file to read in the refined boundary region coordinates
    gs = np.loadtxt('%s/pfs.dat' % folder,dtype=str)
    for i in range(len(gs)):
        params = open('%s/%s' % (folder,gs[i]))
        for l in params:
            #Obtain the x,y,z coordinate of the left edge
            if l.startswith('RefineRegionLeftEdge'):
                le_i = l.split()
                le_i = [float(x) for x in le_i[-3:]]
                le_i = np.array(le_i)
                if i ==0:
                    le = le_i
                else:
                    le = np.vstack((le,le_i))
            #Obtain the x,y,z coordinate of the right edge
            if l.startswith('RefineRegionRightEdge'):
                re_i = l.split()
                re_i = [float(x) for x in re_i[-3:]]
                re_i = np.array(re_i)
                if i ==0:
                    re = re_i
                else:
                    re = np.vstack((re,re_i))

    #Create a dictionary to store the output
    my_storage = {}
    for sto, x in yt.parallel_objects(range(min(numtrees,len(tree))),nprocs, storage = my_storage):
        # Parallel by halo tree in list (most will fail the test)
        sto.result = {}
        hlist = {}
        #This key contains the name of all the roots (both branches and trees) in the arbor
        hlist['uids'] = np.array([])
        hlist['prog_found'] = np.array([])
        hlist['rootlist'] = np.array([])
        uid = tree[x]['uid']
        root = tree[x]
        rootn = '0'
        mlist = np.array([])
        merger_tree = np.array([])
        # Get the initial halo list values for the primary branch and any sub-branches
        hlist,mlist,merger_tree = getleaves(re,le,root,rootn,uid,hlist,mlist,merger_tree,snapstart=0,isroot=True,min_r=min_r,min_mass=min_mass)
        for z in range(depth):
            # Check all nodes in the branch for progenitors. The number of times we do this check
            # is equivalent to the number of times we search for sub-branches
            for y in np.where( hlist['prog_found'] ==0)[0]:
                hlist,mlist,merger_tree = getleaves(re,le,root,hlist['rootlist'][y],hlist['uids'][y],hlist,mlist,merger_tree,snapstart=0,min_r=min_r,min_mass=min_mass)
        sto.result[0] = hlist
        sto.result[1] = mlist
        sto.result[2] = merger_tree
    root = 0
    masses = np.array([])
    hlist_f = {}
    merger_time = {}
    for c, v in sorted(my_storage.items()):
        count = 0
        if len(v[0]['rootlist']) >0:
            for i in range(len(v[0]['rootlist'])):
                iroot = v[0]['rootlist'][i]
                # Renumbers the root of the trees in order of placement in the original tree
                index = iroot.split('_')[1:]
                index.insert(0,str(root))
                index = '_'.join(index)

                # Populates the final halo list with position and radius
                # Keeps track of root mass for sorting
                if len(v[0][iroot]) > 0:
                    count += 1
                    masses = np.append(masses,v[1][i])
                    hlist_f[index] = v[0][iroot]
                    merger_time[index] = int(v[2][i])

                #if yt.is_root():
                #  print(len(masses),len(list(hlist_f.keys())),index,root,v[1][i])
        if count >0:
            root += 1
    # Saves merger timings
    if yt.is_root():
        np.save('%s/merger_time.npy' % folder,merger_time)
    return hlist_f,masses

def add_leaves(leaf,re,le,hlist,curroot,snapstart,min_r=1e-3, min_mass = 1e6):
    x = leaf['position_x'].v.tolist()
    y = leaf['position_y'].v.tolist()
    z = leaf['position_z'].v.tolist()
    pos2 = np.array([x,y,z])
    rvir = leaf['virial_radius'].to('unitary').v.tolist()
    # Performs a check to make sure that the node is in the region and larger than
    # the minimum radius as well as massive enough.
    cmass = leaf['Mvir_all'].to('Msun').v.tolist()
    if (np.sum(pos2 >le[int(leaf['Snap_idx'])])==3)*(np.sum(pos2 <re[int(leaf['Snap_idx'])])==3) and rvir >min_r and cmass > min_mass:
        snap = int(leaf['Snap_idx']) + snapstart
        hlist[curroot][snap] = []
        hlist[curroot][snap].append([x,y,z])
        hlist[curroot][snap].append(rvir)
        hlist[curroot][snap].append(cmass)
    return hlist


def getleaves(re,le,root,curroot,uid,hlist,mlist,merger_tree,snapstart=0,isroot=False,min_r=1e-3, min_mass=1e6):
    rooti = np.array(list(root['tree']))[root['tree','uid'] == uid][0]
    cmass = rooti['Mvir_all'].to('Msun').v.tolist()
    bool1 = rooti['prog','mass'].to('Msun') > min_mass
    # Checks that the root node is large enough and massive enough.
    # This only runs for the root since this will also build the branches
    # from the root. The second run of this code only builds more branches.
    if isroot and rooti['virial_radius'].to('unitary')>min_r and cmass > min_mass:
        merger_tree = np.append(merger_tree,-1)
        mlist = np.append(mlist,cmass)
        hlist['rootlist'] = np.append(hlist['rootlist'],curroot)
        # Notes that this is a new node that has not been checked for progenitors yet.
        hlist['prog_found'] = np.append(hlist['prog_found'],0)
        # Notes the unique ID of the tree we are examining.
        hlist['uids'] = np.append(hlist['uids'],int(rooti['tree','uid'][0]))
        hlist[curroot] = {}
        # Limits leaves to those above the mass limit.
        for leaf in np.array(list(rooti['prog']))[bool1]:
            hlist = add_leaves(leaf,re,le,hlist,curroot,snapstart,min_r=min_r,min_mass=min_mass)
    # Find progenitor nodes with more than one progenitor
    bool2 = rooti['prog','num_prog'][bool1] >1
    branchn = 0
    for coroot in np.array(list(rooti['prog']))[bool1][bool2]:
        # Explores all ancestors but the main branch
        # Same as above except the halo name evolves.
        for i in range(1,len(list(coroot.ancestors))):
            corooti = list(coroot.ancestors)[i]
            cmass = corooti['Mvir_all'].to('Msun').v.tolist()
            if cmass > min_mass and corooti['virial_radius'].to('unitary')>min_r:
                mlist = np.append(mlist,cmass)
                merger_tree = np.append(merger_tree,int(coroot['Snap_idx']) + snapstart)
                curcon = curroot+'_'+'%s' % branchn
                hlist['rootlist'] = np.append(hlist['rootlist'],curcon)
                hlist['prog_found'] = np.append(hlist['prog_found'], 0)
                hlist[curcon] = {}
                hlist['uids'] = np.append(hlist['uids'],int(corooti['tree','uid'][0]))
                bool3 = corooti['prog','mass'].to('Msun') > min_mass
                for leaf in np.array(list(corooti['prog']))[bool3]:
                    hlist = add_leaves(leaf,re,le,hlist,curcon,snapstart,min_r=min_r,min_mass=min_mass)
                branchn += 1
    # Establishes that the progentors for this root have been found.
    # Any sub-sub branches are still un-solved.
    bool4 = hlist['uids'] == uid
    hlist['prog_found'][bool4] = 1
    return hlist,mlist,merger_tree

#--------------------------------------------------------------------
#Main code
trees = 10000
output_name = 'halotree.npy'

#The directory to the folder containing the simulation snapshots
folder = sys.argv[-1]
#The name of the folder containing rockstar outputs
rockfolder = 'rockstar_halos'

#Make pfs_manual.dat file from the rockstar_halos/outputs/scales.txt file (regardless whether
#we have the original pfs.dat or not). 
#This is VERY IMPORTANT because consistent-tree may not use all the rockstar outputs
#because not all of them have halos. Therefore, consistent-tree will follow the index (via Snap_idx)
#from the rockstar_halos/outputs/scales.txt file. Thus, we need to create the pfs_manual.dat
#file so we can match the consistent-tree indices with the snapshot correctly when extracting
#the snapshot properties.
if yt.is_root():
  if os.path.exists('%s/pfs_manual.dat' % folder) == False:
    make_pfs(folder,rockfolder)

#Setting the minimum radius and the minimum mass for the halos
min_r = 5e-4
min_mass = 1e6

#Building the merger history
hlist,masses = makehlist(folder,rockfolder,trees,depth=7,min_r=min_r,min_mass=min_mass)
#Writing out the output
if yt.is_root():
   np.save('%s/%s' % (folder,output_name),hlist)