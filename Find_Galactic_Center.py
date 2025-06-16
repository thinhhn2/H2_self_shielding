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

class Find_Galactic_Center():
    """
    Parameters
    ----------
    ds : yt dataset
        The dataset snapshot of the simulation
    oden : float
        The overdensity threshold to find the center of the galaxy (common value is 2000)
    halo_center : numpy.ndarray
        The center of the DM halo
    halo_rvir : float
        The virial radius of the DM halo
    star_pos : numpy.ndarray (optional)
        The positions of the star particles (stored in star_metadata file)
    star_mass : numpy.ndarray (optional)
        The masses of the star particles (stored in star_metadata file)
    """
    def __init__(self, ds, oden, halo_center, halo_rvir, star_pos = None, star_mass = None):
        self.ds = ds
        self.oden = oden
        self.halo_center = halo_center
        self.halo_rvir = halo_rvir
        self.star_pos = star_pos if star_pos is not None else None
        self.star_mass = star_mass if star_mass is not None else None
    #
    def plot_star_particles(self, center, radius, bins=800, saveplot = False, savedir = None):
        plt.figure(figsize=(22.5,8))
        #
        x = self.star_pos[:,0]
        y = self.star_pos[:,1]
        z = self.star_pos[:,2]
        #
        rel_bool = np.linalg.norm(self.star_pos - center, axis=1) < radius
        rel_x = self.star_pos[:,0][rel_bool]
        rel_y = self.star_pos[:,1][rel_bool]
        rel_z = self.star_pos[:,2][rel_bool]
        #
        ax1 = plt.subplot(1,3,1)
        _ = ax1.hist2d(rel_x, rel_y, bins=bins, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
        ax1.set_xlabel('x (code_length)', fontsize=16)
        ax1.set_ylabel('y (code_length)', fontsize=16)
        ax1.tick_params(axis='x', labelsize=16)
        ax1.tick_params(axis='y', labelsize=16)
        ax1.xaxis.set_minor_locator(AutoMinorLocator())
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax1.grid(which='both')
        ax1.set_xlim(center[0] - radius, center[0] + radius)
        ax1.set_ylim(center[1] - radius, center[1] + radius)
        ax1.set_aspect('equal') 
        circle1 = plt.Circle((center[0], center[1]), radius, color='r', fill=False)
        ax1.add_patch(circle1)
        #
        ax2 = plt.subplot(1,3,2)
        _ = ax2.hist2d(rel_y, rel_z, bins=bins, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
        ax2.set_xlabel('y (code_length)', fontsize=16)
        ax2.set_ylabel('z (code_length)', fontsize=16)
        ax2.tick_params(axis='x', labelsize=16)
        ax2.tick_params(axis='y', labelsize=16)
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.grid(which='both')
        ax2.set_xlim(center[1] - radius, center[1] + radius)
        ax2.set_ylim(center[2] - radius, center[2] + radius)
        ax2.set_aspect('equal')
        circle2 = plt.Circle((center[1], center[2]), radius, color='r', fill=False)
        ax2.add_patch(circle2)
        #
        ax3 = plt.subplot(1,3,3)
        _ = ax3.hist2d(rel_z, rel_x, bins=bins, norm=mpl.colors.LogNorm(), cmap=mpl.cm.viridis)
        ax3.set_xlabel('z (code_length)', fontsize=16)
        ax3.set_ylabel('x (code_length)', fontsize=16)
        ax3.tick_params(axis='x', labelsize=16)
        ax3.tick_params(axis='y', labelsize=16)
        ax3.xaxis.set_minor_locator(AutoMinorLocator())
        ax3.yaxis.set_minor_locator(AutoMinorLocator())
        ax3.grid(which='both')
        ax3.set_xlim(center[2] - radius, center[2] + radius)
        ax3.set_ylim(center[0] - radius, center[0] + radius)
        ax3.set_aspect('equal')
        circle3 = plt.Circle((center[2], center[0]), radius, color='r', fill=False)
        ax3.add_patch(circle3)
        #plt.locator_params(axis='both', nbins=10) 
        plt.tight_layout()
        if saveplot == True:
            plt.savefig(savedir, dpi = 300, bbox_inches='tight')
    #
    def plot_gas_projection(self, center, char_radius, codelength_mode = True, saveplot = False, savedir = None):
        if codelength_mode == False:
            prj_list = ['z','x','y']
            reg = self.ds.sphere(center, (self.halo_rvir, 'code_length'))
            fig = plt.figure(figsize=(9*2,6*2))
            grid = AxesGrid(
                fig,
                (0.075, 0.075, 0.85, 0.85),
                nrows_ncols=(1, 3),
                axes_pad=0.8,
                label_mode="all",
                cbar_pad="2%",
                aspect=False,
            )
            for j in range(3):
                prj = yt.ProjectionPlot(
                    self.ds,
                    prj_list[j],
                    ("gas", "density"),
                    center = center,
                    #width = self.halo_rvir,
                    width = 6*char_radius,
                    weight_field=("gas", "density"),
                    data_source = reg
                )
                prj.set_cmap(("gas", "density"), 'viridis')
                prj.set_font_size(14)
                prj.set_unit(("gas", "density"), "Msun/pc**3")
                prj.set_zlim(("gas", "density"), 1e-4, 1e4)
                prj.annotate_sphere(
                    center,
                    radius=char_radius,
                    coord_system="data"
                )
                plot = prj.plots[("gas", "density")]
                plot.figure = fig
                plot.axes = grid[j].axes
                prj.render()
            if saveplot == True:
                plt.savefig(savedir, dpi = 300, bbox_inches='tight')
        elif codelength_mode == True:
            prj_list = ['z','x','y']
            reg = self.ds.sphere(center, (self.halo_rvir, 'code_length'))
            for j in range(3):
                prj = yt.ProjectionPlot(
                    self.ds,
                    prj_list[j],
                    ("gas", "density"),
                    center = center,
                    #width = self.halo_rvir,
                    width = 6*char_radius,
                    weight_field=("gas", "density"),
                    origin = 'native',
                    data_source = reg
                )
                prj.set_cmap(("gas", "density"), 'viridis')
                prj.set_font_size(14)
                prj.set_unit(("gas", "density"), "Msun/pc**3")
                prj.set_axes_unit("code_length")
                prj.set_zlim(("gas", "density"), 1e-4, 1e4)
                prj.annotate_sphere(
                    center,
                    radius=char_radius,
                    coord_system="data"
                )
                prj.show()
    #
    def univDen(self):
        # Hubble constant
        H0 = self.ds.hubble_constant * 100 * u.km/u.s/u.Mpc
        H = H0**2 * (self.ds.omega_matter*(1 + self.ds.current_redshift)**3 + self.ds.omega_lambda)  # Technically H^2
        G = 6.67e-11 * u.m**3/u.s**2/u.kg
        # Density of the universe
        den = (3*H/(8*np.pi*G)).to("kg/m**3") / u.kg * u.m**3
        return den.value
    #
    def Find_Com_and_virRad(self,initial_gal_com_manual = False, initial_gal_com = None):
        #Obtain the gas + star info in the halo
        reg = self.ds.sphere(self.halo_center, (self.halo_rvir, 'code_length'))
        gas_mass = reg['gas', 'cell_mass'].in_units("kg").v.tolist()
        gas_x = reg[("gas","x")].in_units("m").v.tolist()
        gas_y = reg[("gas","y")].in_units("m").v.tolist()
        gas_z = reg[("gas","z")].in_units("m").v.tolist()
        gas_pos = np.array([gas_x, gas_y, gas_z]).T
        #
        star_pos_si = (self.star_pos*self.ds.units.code_length).to('m').v
        star_mass_si = (self.star_mass*self.ds.units.Msun).to('kg').v
        pos = np.concatenate([star_pos_si, gas_pos], axis=0)
        mass = np.append(star_mass_si, gas_mass)
        if initial_gal_com_manual == True:
            initial_gal_com = initial_gal_com
        else:
            try:
                zoomin = np.linalg.norm(self.star_pos - self.halo_center, axis=1) < 0.2*self.halo_rvir
                initial_gal_com = np.average(star_pos_si[zoomin], weights=star_mass_si[zoomin], axis=0)
            except:
                initial_gal_com = np.average(star_pos_si, weights=star_mass_si, axis=0)
        #
        halo_rvir_si = (self.halo_rvir*self.ds.units.code_length).to('m').v
        #Make sure the unit is in kg and m
        extension = 0.1 #default is 0.1
        virRad = 0.01*halo_rvir_si #default is 0.05
        com = initial_gal_com
        fden = np.inf
        while extension >= 0.01:
            virRad_new = virRad + extension*halo_rvir_si
            r = np.linalg.norm(pos - com, axis=1)
            pos_in = pos[r < virRad_new]
            mass_in = mass[r < virRad_new]
            den = np.sum(mass_in) / (4/3 * np.pi * virRad_new**3)
            uden = self.univDen()
            fden = den/uden
            com_new = np.average(pos_in, weights=mass_in, axis=0)
            if fden < self.oden:
                extension -= 0.01
            else:
                com = com_new
                virRad = virRad_new
        return com, virRad
        