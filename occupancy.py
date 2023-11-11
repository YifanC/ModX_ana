import numpy as np
import h5py
import h5flow
import matplotlib.pyplot as plt
import sys

x_boundaries = np.array([-30.431, 30.431])
y_boundaries = np.array([-61.8543, 61.8543])
z_boundaries = np.array([-30.8163, 30.8163])

common_path = '/global/cfs/cdirs/dune/www/data/ModuleX/'
runlist_path = 'runlist/'
runlist_name = 'runlist_flow_0-5_Efield_1_copy.txt'
output_path = 'plots/'
output_prefix = 'occupancy_'

f_runlist = open(f'{common_path}{runlist_path}{runlist_name}', 'r')
names = f_runlist.read().splitlines()

for i_f, f_name in enumerate(names):
    if i_f >1:
        break
    print(f_name)
    f_label = f_name[-24:-3]
    out = f"{output_path}{output_prefix}{f_label}"
    f_manager = h5flow.data.H5FlowDataManager(f_name, 'r')

    PromptHits = f_manager["charge/events", "charge/calib_prompt_hits"]

    tpc1_mask = (PromptHits['x']<0)
    arr_z_tpc1 = PromptHits['z'][tpc1_mask]
    arr_y_tpc1 = PromptHits['y'][tpc1_mask]

    tpc2_mask = (PromptHits['x']>0)
    arr_z_tpc2 = PromptHits['z'][tpc2_mask]
    arr_y_tpc2 = PromptHits['y'][tpc2_mask]


    ########
    # xmin
    ########
    h_tpc1_count, zedges, yedges = np.histogram2d(arr_z_tpc1, arr_y_tpc1, bins=[140,280], range=[[z_boundaries[0],z_boundaries[1]],[y_boundaries[0],y_boundaries[1]]])
    Z, Y = np.meshgrid(zedges, yedges)
    plt.figure(figsize=(6,10))
    # c_tpc1_count = plt.pcolormesh(Z, Y, h_tpc1_count.T)
    c_tpc1_count = plt.pcolormesh(Z, Y, h_tpc1_count.T, vmax=800)
    plt.colorbar(c_tpc1_count)
    plt.xlabel("z [cm]")
    plt.ylabel("y [cm]")
    plt.title("Occupancy in TPC 1")
    plt.margins(0.2)
    plt.set_cmap("Greens")
    plt.savefig(f"{output_path}{output_prefix}TPC1_{f_label}.pdf")
    plt.show()

    ########
    # xmax
    ########
    h_tpc2_count, zedges, yedges = np.histogram2d(arr_z_tpc2, arr_y_tpc2, bins=[140,280], range=[[z_boundaries[0],z_boundaries[1]],[y_boundaries[0],y_boundaries[1]]])
    Z, Y = np.meshgrid(zedges, yedges)
    plt.figure(figsize=(6,10))
    c_tpc2_count = plt.pcolormesh(Z, Y, h_tpc2_count.T, vmax=800)
    plt.colorbar(c_tpc2_count)
    plt.xlabel("z [cm]")
    plt.ylabel("y [cm]")
    plt.title("Occupancy in TPC 2")
    plt.margins(0.2)
    plt.set_cmap("Greens")
    plt.savefig(f"{output_path}{output_prefix}TPC2_{f_label}.pdf")
    plt.show()

    ########
    # z
    ########
    h_z_count, zedges, yedges = np.histogram2d(PromptHits['x'], PromptHits['y'], bins=[140,280], range=[[z_boundaries[0],z_boundaries[1]],[y_boundaries[0],y_boundaries[1]]])
    Z, Y = np.meshgrid(zedges, yedges)
    plt.figure(figsize=(6,10))
    c_z_count = plt.pcolormesh(Z, Y, h_z_count.T, vmax=800)
    plt.colorbar(c_z_count)
    plt.xlabel("x [cm]")
    plt.ylabel("y [cm]")
    plt.title("Occupancy (z projection)")
    plt.margins(0.2)
    plt.set_cmap("Greens")
    plt.savefig(f"{output_path}{output_prefix}z_proj_{f_label}.pdf")
    plt.show()

    ########
    # y
    ########
    h_y_count, zedges, yedges = np.histogram2d(PromptHits['x'], PromptHits['z'], bins=[140,140], range=[[z_boundaries[0],z_boundaries[1]],[z_boundaries[0],z_boundaries[1]]])
    Z, Y = np.meshgrid(zedges, yedges)
    plt.figure(figsize=(5,4))
    c_y_count = plt.pcolormesh(Z, Y, h_y_count.T, vmax=800)
    plt.colorbar(c_y_count)
    plt.xlabel("x [cm]")
    plt.ylabel("z [cm]")
    plt.title("Occupancy (y projection)")
    plt.margins(0.2)
    plt.set_cmap("Greens")
    plt.savefig(f"{output_path}{output_prefix}y_proj_{f_label}.pdf")
    plt.show()
