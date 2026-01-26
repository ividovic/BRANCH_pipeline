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



starting_number = 38
row_number = 1
sorta = "V"
saveResults = True
voxel_size = 0.02 #1cm or 0.02
NOISE_BOUND = voxel_size #should be same as voxel size
full_path_merged = "/home/user/BRANCH_v2/images/asus/Merged/distinctive_merged/voxel_002_mobdok/filtrated_good_models_teaser"

for i in range(starting_number,184): 

    if(i<10):
        tree="tree_"+ str(row_number)+ "_" + sorta + "_" + "000" + str(i)
    elif(i>=10 and i <= 99):
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "00" + str(i)
    else:
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "0" + str(i)

    print("Working on image: " + tree)  

    pcd= o3d.io.read_point_cloud(full_path_merged + "/" + tree + "_teaser.ply")
    o3d.visualization.draw_geometries([pcd], window_name = "Teaser merged: "+ str(tree))
    if (len(pcd.points) > 0):
        points = np.asarray(pcd.points) 
        colors = np.asarray(pcd.colors) 
        labels = color_to_label(colors)
        save_Labels(full_path_merged, voxel_size, tree, labels)
    #print("labels: ",labels)


    
