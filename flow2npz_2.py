import numpy as np

import h5py

import h5flow
from h5flow.data import dereference

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

import sys

import difflib

x_boundaries = np.array([-30.431, 30.431])
y_boundaries = np.array([-61.8543, 61.8543])
z_boundaries = np.array([-30.8163, 30.8163])

edge_threshold = 4
ct_hits_threshold = 10
crazy_n_hits_threshold = 5000
seg_n_hits = 30

def divide_closest(n, d):
    return (n + d // 2) // d

def hit_xmax_edge(hit_x, hit_y, hit_z):
    if (np.max(hit_x) > np.max(x_boundaries) - edge_threshold) and (np.max(hit_x) < np.max(x_boundaries) + edge_threshold):
        edge_idx = np.argmax(hit_x)
        return hit_x[edge_idx], hit_y[edge_idx], hit_z[edge_idx]
    else:
        return False

def hit_xmin_edge(hit_x, hit_y, hit_z):
    if (np.min(hit_x) > np.min(x_boundaries) - edge_threshold) and (np.min(hit_x) < np.min(x_boundaries) + edge_threshold):
        edge_idx = np.argmin(hit_x)
        return hit_x[edge_idx], hit_y[edge_idx], hit_z[edge_idx]
    else:
        return False

def hit_ymax_edge(hit_x, hit_y, hit_z):
    if (np.max(hit_y) > np.max(y_boundaries) - edge_threshold) and (np.max(hit_y) < np.max(y_boundaries) + edge_threshold):
        edge_idx = np.argmax(hit_y)
        return hit_x[edge_idx], hit_y[edge_idx], hit_z[edge_idx]
    else:
        return False

def hit_ymin_edge(hit_x, hit_y, hit_z):
    if (np.min(hit_y) > np.min(y_boundaries) - edge_threshold) and (np.min(hit_y) < np.min(y_boundaries) + edge_threshold):
        edge_idx = np.argmin(hit_y)
        return hit_x[edge_idx], hit_y[edge_idx], hit_z[edge_idx]
    else:
        return False

def hit_zmax_edge(hit_x, hit_y, hit_z):
    if (np.max(hit_z) > np.max(z_boundaries) - edge_threshold) and (np.max(hit_z) < np.max(z_boundaries) + edge_threshold): 
        edge_idx = np.argmax(hit_z)
        return hit_x[edge_idx], hit_y[edge_idx], hit_z[edge_idx]
    else:
        return False
    
def hit_zmin_edge(hit_x, hit_y, hit_z):
    if (np.min(hit_z) > np.min(z_boundaries) - edge_threshold) and (np.min(hit_z) < np.min(z_boundaries) + edge_threshold): 
        edge_idx = np.argmin(hit_z)
        return hit_x[edge_idx], hit_y[edge_idx], hit_z[edge_idx]
    else:
        return False

def hit_cathode_xplus(hit_x, hit_y, hit_z):
    if len(hit_x[hit_x>0])> 0:
        if np.min(hit_x[hit_x>0]) < edge_threshold:
            edge_idx = np.argmin(hit_x[hit_x>0])
            return hit_x[hit_x>0][edge_idx], hit_y[hit_x>0][edge_idx], hit_z[hit_x>0][edge_idx]
    else:
        return False

def hit_cathode_xminus(hit_x, hit_y, hit_z):
    if len(hit_x[hit_x<0])> 0:
        if np.max(hit_x[hit_x<0]) > -edge_threshold:
            edge_idx = np.argmax(hit_x[hit_x<0])
            return hit_x[hit_x<0][edge_idx], hit_y[hit_x<0][edge_idx], hit_z[hit_x<0][edge_idx]
    else:
        return False

def out_time(hit_x, hit_y, hit_z):
    if (np.max(hit_x) >= np.max(x_boundaries) + edge_threshold) or (np.min(hit_x) <= np.min(x_boundaries) - edge_threshold):
        return True
    
    if hit_xmax_edge(hit_x, hit_y, hit_z):
        hit_xmax_x, hit_xmax_y, hit_xmax_z = hit_xmax_edge(hit_x, hit_y, hit_z)
        if hit_xmax_y < np.max(y_boundaries)/2:
            return True        
    return False

def hit_ct_end(hit_x, hit_y, hit_z):
    ct_end = []
    end1 = np.argmax(hit_x)
    end2 = np.argmin(hit_x)
    ct_end.append([hit_x[end1], hit_y[end1], hit_z[end1]])
    ct_end.append([hit_x[end2], hit_y[end2], hit_z[end2]])
    return ct_end

common_path = '/global/cfs/cdirs/dune/www/data/ModuleX/'
runlist_path = 'runlist/'
runlist_name = 'runlist_flow_0-5_Efield_2.txt'
output_path = 'analysis_data/analysis_0-5_Efield_2/'
output_prefix = 'hit_seg_edge_'

f_runlist = open(f'{common_path}{runlist_path}{runlist_name}', 'r')
#names = f_runlist.readlines() # will give a new line
names = f_runlist.read().splitlines()

for f_name in names:
    print(f_name)
    f_label = f_name[-24:-3]
    outfile = f"{common_path}{output_path}{output_prefix}{f_label}"

    f_manager = h5flow.data.H5FlowDataManager(f_name, 'r')

    edge_xmax_hits = []
    edge_xmin_hits = []
    edge_ymax_hits = []
    edge_ymin_hits = []
    edge_zmax_hits = []
    edge_zmin_hits = []
    edge_cathode_xplus_hits = []
    edge_cathode_xminus_hits = []
    
    evt_crossing_track = []
    
    ct_fit_length = []
    ct_PCA_mean = []
    ct_PCA_vec = []
    ct_PCA_variance = []
    ct_PCA_singular_values = []
    ct_pts_density = []
    
    seg_fit_length = []
    seg_PCA_mean = []
    seg_PCA_vec = []
    seg_PCA_variance = []
    seg_PCA_singular_values = []
    seg_pts_density = []
    
    seg_cos_angle = []
    seg_angle = []
    seg_joint = []
    
    ct_ends = []

    f_n_evt = f_manager["charge/events/data"]
    
    for i_evt in f_n_evt['id']:
        if f_n_evt['nhit'][i_evt] < ct_hits_threshold:
            continue
        if f_n_evt['nhit'][i_evt] > crazy_n_hits_threshold:
            continue
    
        PromptHits_ev = f_manager["charge/events", "charge/calib_prompt_hits", i_evt]
        this_hit_x, this_hit_y, this_hit_z = PromptHits_ev.data['x'].flatten(), PromptHits_ev.data['y'].flatten(), PromptHits_ev.data['z'].flatten()
        nan_mask = np.isfinite(this_hit_x) & np.isfinite(this_hit_y) & np.isfinite(this_hit_z)
        hit_x = this_hit_x[nan_mask]
        hit_y = this_hit_y[nan_mask]
        hit_z = this_hit_z[nan_mask]
        pts = np.array([hit_x, hit_y, hit_z]).T
    
        db = DBSCAN(eps=5, min_samples=3).fit(pts)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
    
        if hit_xmax_edge(hit_x, hit_y, hit_z) and hit_xmin_edge(hit_x, hit_y, hit_z):
            evt_crossing_track.append(i_evt)
    
        for i_ct in range(n_clusters_):
            ct_mask = labels == i_ct
            hit_x_ct = hit_x[ct_mask]
            hit_y_ct = hit_y[ct_mask]
            hit_z_ct = hit_z[ct_mask]
    
            n_hit_ct = np.count_nonzero(ct_mask)
    
            if out_time(hit_x_ct, hit_y_ct, hit_z_ct):
                continue
            if n_hit_ct < ct_hits_threshold:
                continue
    
            #####################
            # Fill edge most hits
            #####################
            if hit_xmax_edge(hit_x_ct, hit_y_ct, hit_z_ct):
                edge_xmax_hits.append(hit_xmax_edge(hit_x_ct, hit_y_ct, hit_z_ct))
            if hit_xmin_edge(hit_x_ct, hit_y_ct, hit_z_ct):
                edge_xmin_hits.append(hit_xmin_edge(hit_x_ct, hit_y_ct, hit_z_ct))
            if hit_ymax_edge(hit_x_ct, hit_y_ct, hit_z_ct):
                edge_ymax_hits.append(hit_ymax_edge(hit_x_ct, hit_y_ct, hit_z_ct))
            if hit_ymin_edge(hit_x_ct, hit_y_ct, hit_z_ct):
                edge_ymin_hits.append(hit_ymin_edge(hit_x_ct, hit_y_ct, hit_z_ct))
            if hit_zmax_edge(hit_x_ct, hit_y_ct, hit_z_ct):
                edge_zmax_hits.append(hit_zmax_edge(hit_x_ct, hit_y_ct, hit_z_ct))
            if hit_zmin_edge(hit_x_ct, hit_y_ct, hit_z_ct):
                edge_zmin_hits.append(hit_zmin_edge(hit_x_ct, hit_y_ct, hit_z_ct))
            if hit_cathode_xplus(hit_x_ct, hit_y_ct, hit_z_ct):
                edge_cathode_xplus_hits.append(hit_cathode_xplus(hit_x_ct, hit_y_ct, hit_z_ct))
            if hit_cathode_xminus(hit_x_ct, hit_y_ct, hit_z_ct):
                edge_cathode_xminus_hits.append(hit_cathode_xminus(hit_x_ct, hit_y_ct, hit_z_ct))
    
            #####################
            # Fill end of cluster hits
            #####################
            this_ct_ends = hit_ct_end(hit_x_ct, hit_y_ct, hit_z_ct)
            for i_end in range(len(this_ct_ends)):
                ct_ends.append(this_ct_ends[i_end])           
                    
            #####################
            # PCA
            #####################
            ct_pts = np.array([hit_x_ct, hit_y_ct, hit_z_ct]).T
            pca = PCA(n_components=2)
            pca.fit(ct_pts)
            
            centre = pca.mean_
            zmax_idx = np.argmax(hit_z_ct)
            zmin_idx = np.argmin(hit_z_ct)
            length = np.linalg.norm([(hit_x_ct[zmax_idx]-hit_x_ct[zmin_idx]),(hit_y_ct[zmax_idx]-hit_y_ct[zmin_idx]),(hit_z_ct[zmax_idx]-hit_z_ct[zmin_idx])]) # estimated length, line assumption
            end1 = centre + 0.5 * length * pca.components_[0]
            end2 = centre - 0.5 * length * pca.components_[0]
           
            centre = pca.mean_
            zmax_idx = np.argmax(hit_z_ct)
            zmin_idx = np.argmin(hit_z_ct)
            length_z = np.linalg.norm([(hit_x_ct[zmax_idx]-hit_x_ct[zmin_idx]),(hit_y_ct[zmax_idx]-hit_y_ct[zmin_idx]),(hit_z_ct[zmax_idx]-hit_z_ct[zmin_idx])]) # estimated length, line assumption
            ymax_idx = np.argmax(hit_y_ct)
            ymin_idx = np.argmin(hit_y_ct)
            length_y = np.linalg.norm([(hit_x_ct[ymax_idx]-hit_x_ct[ymin_idx]),(hit_y_ct[ymax_idx]-hit_y_ct[ymin_idx]),(hit_z_ct[ymax_idx]-hit_z_ct[ymin_idx])]) # estimated length, line assumption
            xmax_idx = np.argmax(hit_x_ct)
            xmin_idx = np.argmin(hit_x_ct)
            length_x = np.linalg.norm([(hit_x_ct[xmax_idx]-hit_x_ct[xmin_idx]),(hit_y_ct[xmax_idx]-hit_y_ct[xmin_idx]),(hit_z_ct[xmax_idx]-hit_z_ct[xmin_idx])]) # estimated length, line assumption
            length = np.max([length_x, length_y, length_z])
            end1 = centre + 0.5 * length * pca.components_[0]
            end2 = centre - 0.5 * length * pca.components_[0]
    
            ct_fit_length.append(length)
            ct_PCA_mean.append(centre)
            ct_PCA_vec.append(pca.components_[0])
            ct_PCA_variance.append(pca.explained_variance_)
            ct_PCA_singular_values.append(pca.singular_values_)
            ct_pts_density.append(n_hit_ct/length)
    
            ## fit segments
            sort_x_idx = np.argsort(hit_x_ct)
            ct_pts_sort_x = ct_pts[sort_x_idx]
            for i_seg in range(divide_closest(n_hit_ct, seg_n_hits)):
                start_idx = seg_n_hits * i_seg
                if i_seg == (divide_closest(n_hit_ct, seg_n_hits) - 1):
                    end_idx = max(seg_n_hits * (i_seg+1), len(ct_pts_sort_x))
                else:
                    end_idx = seg_n_hits * (i_seg+1)
    
                seg_pts = ct_pts_sort_x[start_idx: end_idx]
                pca_seg = PCA(n_components=2)
                pca_seg.fit(seg_pts)
    
                seg_centre = pca_seg.mean_
                seg_zmax_idx = np.argmax(seg_pts[:,2])
                seg_zmin_idx = np.argmin(seg_pts[:,2])
                seg_length_z = np.linalg.norm([(seg_pts[:,0][seg_zmax_idx]-seg_pts[:,0][seg_zmin_idx]),(seg_pts[:,1][seg_zmax_idx]-seg_pts[:,1][seg_zmin_idx]),(seg_pts[:,2][seg_zmax_idx]-seg_pts[:,2][seg_zmin_idx])]) # estimated length, line assumption
                seg_ymax_idx = np.argmax(seg_pts[:,1])
                seg_ymin_idx = np.argmin(seg_pts[:,1])
                seg_length_y = np.linalg.norm([(seg_pts[:,0][seg_ymax_idx]-seg_pts[:,0][seg_ymin_idx]),(seg_pts[:,1][seg_ymax_idx]-seg_pts[:,1][seg_ymin_idx]),(seg_pts[:,2][seg_ymax_idx]-seg_pts[:,2][seg_ymin_idx])]) # estimated length, line assumption
                seg_xmax_idx = np.argmax(seg_pts[:,0])
                seg_xmin_idx = np.argmin(seg_pts[:,0])
                seg_length_x = np.linalg.norm([(seg_pts[:,0][seg_xmax_idx]-seg_pts[:,0][seg_xmin_idx]),(seg_pts[:,1][seg_xmax_idx]-seg_pts[:,1][seg_xmin_idx]),(seg_pts[:,2][seg_xmax_idx]-seg_pts[:,2][seg_xmin_idx])]) # estimated length, line assumption
                seg_length = np.max([length_x, length_y, length_z])
                seg_end1 = seg_centre + 0.5 * seg_length_z * pca_seg.components_[0]
                seg_end2 = seg_centre - 0.5 * seg_length_z * pca_seg.components_[0]
    
                seg_fit_length.append(seg_length)
                seg_PCA_mean.append(seg_centre)
                seg_PCA_vec.append(pca_seg.components_[0])
                seg_PCA_variance.append(pca_seg.explained_variance_)
                seg_PCA_singular_values.append(pca_seg.singular_values_)
                seg_pts_density.append(len(seg_pts)/seg_length)
    
                if i_seg > 0:
                    cos_angle = abs(np.sum(seg_PCA_vec[i_seg] * seg_PCA_vec[i_seg-1])/(np.linalg.norm(seg_PCA_vec[i_seg])*np.linalg.norm(seg_PCA_vec[i_seg-1])))
                    angle = np.arccos(cos_angle)/np.pi * 180
                    # angle = np.min([np.arccos(cos_angle)/np.pi * 180, 180 - np.arccos(cos_angle)/np.pi * 180])
                    seg_cos_angle.append(cos_angle)
                    seg_angle.append(angle)
                    seg_joint.append((seg_PCA_mean[i_seg]+seg_PCA_mean[i_seg-1])/2)
    
    edge_xmax_hits = np.array(edge_xmax_hits)
    edge_xmin_hits = np.array(edge_xmin_hits)
    edge_ymax_hits = np.array(edge_ymax_hits)
    edge_ymin_hits = np.array(edge_ymin_hits)
    edge_zmax_hits = np.array(edge_zmax_hits)
    edge_zmin_hits = np.array(edge_zmin_hits)
    edge_cathode_xplus_hits = np.array(edge_cathode_xplus_hits)
    edge_cathode_xminus_hits = np.array(edge_cathode_xminus_hits)
    
    evt_crossing_track = np.array(evt_crossing_track)
    
    ct_fit_length = np.array(ct_fit_length)
    ct_PCA_mean = np.array(ct_PCA_mean)
    ct_PCA_vec = np.array(ct_PCA_vec)
    ct_PCA_variance = np.array(ct_PCA_variance)
    ct_PCA_singular_values = np.array(ct_PCA_singular_values)
    ct_pts_density = np.array(ct_pts_density)
    
    seg_fit_length = np.array(seg_fit_length)
    seg_PCA_mean = np.array(seg_PCA_mean)
    seg_PCA_vec = np.array(seg_PCA_vec)
    seg_PCA_variance = np.array(seg_PCA_variance)
    seg_PCA_singular_values = np.array(seg_PCA_singular_values)
    seg_pts_density = np.array(seg_pts_density)
    
    seg_cos_angle = np.array(seg_cos_angle)
    seg_angle = np.array(seg_angle)
    seg_joint = np.array(seg_joint)
    
    ct_ends = np.array(ct_ends)
    
    np.savez(outfile,
        edge_xmax_hits = edge_xmax_hits,
        edge_xmin_hits = edge_xmin_hits,
        edge_ymax_hits = edge_ymax_hits,
        edge_ymin_hits = edge_ymin_hits,
        edge_zmax_hits = edge_zmax_hits,
        edge_zmin_hits = edge_zmin_hits,
        edge_cathode_xplus_hits = edge_cathode_xplus_hits,
        edge_cathode_xminus_hits = edge_cathode_xminus_hits,
        evt_crossing_track = evt_crossing_track,
        ct_fit_length = ct_fit_length,
        ct_PCA_mean = ct_PCA_mean,
        ct_PCA_vec = ct_PCA_vec,
        ct_PCA_variance = ct_PCA_variance,
        ct_PCA_singular_values = ct_PCA_singular_values,
        ct_pts_density = ct_pts_density,
        seg_fit_length = seg_fit_length,
        seg_PCA_mean = seg_PCA_mean,
        seg_PCA_vec = seg_PCA_vec,
        seg_PCA_variance = seg_PCA_variance,
        seg_PCA_singular_values = seg_PCA_singular_values,
        seg_pts_density = seg_pts_density,
        seg_cos_angle = seg_cos_angle,
        seg_angle = seg_angle,
        seg_joint = seg_joint,
        ct_ends = ct_ends)
