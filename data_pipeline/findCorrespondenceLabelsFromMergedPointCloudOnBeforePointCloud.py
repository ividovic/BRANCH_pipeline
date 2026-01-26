import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import pandas as pd
import open3d as o3d
import numpy as np
import os
import csv
import os.path
import ast


def color_to_label(colors):
        """
        Convert the RGB color values into corresponding labels.
        """
        labels = np.zeros((colors.shape[0]), dtype=np.int32)
        
        red_blue_mask = (colors[:, 0] == 1) & (colors[:, 2] == 1)
        yellow_mask = (colors[:, 0] == 1) & (colors[:, 1] == 1) 
        
        labels[red_blue_mask] = 0 # Red/blue is label 0
        labels[yellow_mask] = 1 # Yellow is label 1
        
        return labels

def save_Labels(full_path_merged, voxel_size, tree, labels):

    # Define the file path for CSV

    csv_file_path = os.path.join(full_path_merged + "/labels_teaser.csv") 
    make_dirs = full_path_merged 

    os.makedirs(make_dirs, exist_ok=True)

    fields = ['voxel_size', 'tree', 'labels']
    full_labels = labels.tolist()
    data = [voxel_size, tree, full_labels]
    # csvwriter.writerow(data)
    # data = [voxel_size, tree, labels]
    
    if(os.path.exists(csv_file_path)):
        with open(csv_file_path, 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
    else:
        with open(csv_file_path, 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(data )


def color_pcd_before(pcd_before, labels_before):
    colors = np.asarray(pcd_before.colors) 
    for i in range(0, len(colors)):
        if(labels_before[i] == 1):
            pcd_before.colors[i][0] = 1
            pcd_before.colors[i][1] = 1
            pcd_before.colors[i][2] = 0


    o3d.visualization.draw_geometries([pcd_before], window_name = "Teaser Before with labels: "+ str(tree))

     


starting_number = 0
row_number = 1
sorta = "V"
saveResults = True
voxel_size = 0.02 #1cm or 0.02
NOISE_BOUND = voxel_size #should be same as voxel size
full_path_merged = "/home/user/BRANCH_v2/images/asus/Merged/distinctive_merged/voxel_002_mobdok/filtrated_good_models_teaser"
full_path_before = "/home/user/BRANCH_v2/images/asus/B/E/filtered_good_models_teaser_before"
counter_forLabels = 0

for i in range(starting_number,184): 

    tree = round(i)
    tree = f"tree_{row_number}_{sorta}_{str(tree).zfill(4)}"

    print("Working on image: " + tree)  

    pcd_merge = o3d.io.read_point_cloud(full_path_merged + "/" + tree + "_teaser.ply")
    pcd_before = o3d.io.read_point_cloud(full_path_before + "/" + tree + "_merged_teaser.ply")
    

    #target - fixed cloud (before)
    #source - rotated pcd (after)
    # in theory -> target is fixed and source is added on top of target    
  

    # filtered_cloud.select_by_index(partOfGrass, invert = False)
    if (len(pcd_merge.points) > 0):
        points = np.asarray(pcd_merge.points) 
        colors = np.asarray(pcd_merge.colors) 
        labels = color_to_label(colors)
        labels_before = labels[0:len(pcd_before.points)]
        #o3d.visualization.draw_geometries([pcd_merge], window_name = "Teaser merged: "+ str(tree))
        color_pcd_before(pcd_before, labels_before)
        #save_Labels(full_path_before, voxel_size, tree, labels)





        # check saved one:
        pcd_before = o3d.io.read_point_cloud(full_path_before + "/" + tree + "_merged_teaser.ply")
        df = pd.read_csv(full_path_before + "/labels_teaser.csv")

        labels_str = df['labels'].iloc[counter_forLabels]  # Get the string like "[0, 1, 0, 1]"
        labels_list = ast.literal_eval(labels_str)  # Convert to actual Python list

        color_pcd_before(pcd_before, labels_list[0:len(pcd_before.points)])
        counter_forLabels = counter_forLabels + 1