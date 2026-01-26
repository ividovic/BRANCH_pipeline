## Author: Jana Dukić
## 3.1.2025. Writing code for visually inspecting and testing BranchVol2_RemoveGrass.py

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
import os.path
import open3d as o3d
import shutil





starting_number = 38
row_number = 1                  # There were from 2 rows images taken: 1 and 5
pear_sort = "V"   
After = False 



for i in range(starting_number,185): 

    if(i<10):
        tree="tree_"+ str(row_number)+ "_" + pear_sort + "_" + "000" + str(i)
    elif(i>=10 and i <= 99):
        tree="tree_" + str(row_number)+ "_" + pear_sort + "_" + "00" + str(i)
    else:
        tree="tree_" + str(row_number)+ "_" + pear_sort + "_" + "0" + str(i)

    if (After):
        B_A = "A"
    else:
        B_A = "B"

    path_to_tree_folder = "/home/user/BRANCH_v2/images/asus/" + B_A +"/E/" + tree 
    full_path = path_to_tree_folder +  "/Filtered__noGrass/pc/"  #/pc_testing/"
    lst = os.listdir(full_path) 
    numberOfImagesOfOneTree = len(lst)
    print(numberOfImagesOfOneTree)
    

    for imageNumber in range(0, numberOfImagesOfOneTree):

        point_cloud = o3d.io.read_point_cloud(full_path+str(imageNumber)+ ".ply")
        o3d.visualization.draw_geometries([point_cloud], window_name = "SAVED PC ---- FOR TESTING: " + tree + ", image: "+ str(imageNumber))
