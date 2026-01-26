#Teaser https://github.com/esteimle/teaser
#https://www.open3d.org/docs/0.8.0/tutorial/Advanced/pointcloud_outlier_removal.html
#https://stackoverflow.com/questions/70160183/how-can-i-align-register-two-meshes-in-open3d-python 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import pandas as pd
import open3d as o3d
import cv2
import numpy as np
import copy
import json
import os
import csv
import os.path
import teaserpp_python 
from timeit import default_timer as timer
import copy
import time
from sklearn.neighbors import KDTree
import shutil

#Asus camera
FX_RGB = 570.3422241210938
FY_RGB = 570.3422241210938
CX_RGB = 319.5
CY_RGB = 239.5

FX_DEPTH = 570.3422241210938
FY_DEPTH = 570.3422241210938
CX_DEPTH = 314.5
CY_DEPTH = 235.5
camera_depth=o3d.camera.PinholeCameraIntrinsic(640,480,FX_DEPTH,FX_DEPTH,CX_DEPTH,CY_DEPTH)

def display_inlier_outlier(cloud, ind, window_name):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    #print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], window_name = window_name)

def preprocess_point_cloud(pcd, voxel_size): #point cloud, voxel (volumetric pixel)
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size) #sets to that voxel dimension (1cm set)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)) #specify parameters for hybrid knn and radial search

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)) #function that does a "fast point feature histogram" for a point cloud
    return pcd_down, pcd_fpfh

#Global registration
def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 10 #1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    start = time.time()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    stop = time.time()
    rezult_time = start-stop
    return [result, rezult_time]

def draw_registration_result(source, target, transformation, tree, imageNumber, window_name):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], window_name = window_name)
    return [source_temp, target_temp]

#Refine registration
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 10 #0.1 #0.01
    #print(":: Point-to-point ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    source.estimate_normals() #needed for TransformationEstimationPointToPlane
    target.estimate_normals() #needed for TransformationEstimationPointToPlane
    start = timer()
    result = o3d.pipelines.registration.registration_icp(source, target, distance_threshold, result_ransac.transformation, o3d.pipelines.registration.TransformationEstimationPointToPoint(),criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=10000))
    end = timer()
    return result, end-start

def save_recontructed_PCD(merged_pcd, teaser_or_icp, treeData,full_path, voxel002):
    if(voxel002):
        destination_path = full_path + "/reconstruction_Voxel002/"
    else:
        destination_path = full_path + "/reconstruction/"
    os.makedirs(destination_path, exist_ok=True)
    name = treeData + "_merged_" + teaser_or_icp+".ply"
    o3d.io.write_point_cloud(destination_path + "/" + name, merged_pcd)
    print("Saved merged point cloud")

def save_timeNeededForTeaser(tree, voxelising_time_list,correspondence_time_list,teaser_time_list,sum_TeaserTime_list):
    destination_path = "/home/user/BRANCH_v2/images/asus/" + B_A +"/E/" 
    csv_file_path = os.path.join(destination_path, "MDPI_times_"+B_A+".csv") 
    fields = ['voxel_size', 'tree', 'voxelisingANDfpfh_time', 'correspondence_time', 'teaser_time', 'sum_TeaserTime']
    data = [voxel_size, tree,  voxelising_time_list, correspondence_time_list, teaser_time_list, sum_TeaserTime_list ]
    if(os.path.exists(csv_file_path)):
        #print("wrote")
        with open(csv_file_path, 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
    else:
        #print("wrote_first")
        with open(csv_file_path, 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(data )


def rotX(theta):
    Rx = [[1, 0 , 0,0], [0 , np.cos(theta), -np.sin(theta),0], [0, np.sin(theta), np.cos(theta),0],[0, 0, 0, 1]]
    return Rx

def rotY(theta):
    Ry = [[np.cos(theta), 0 , np.sin(theta),0], [0 , 1, 0,0], [ -np.sin(theta),0, np.cos(theta),0],[0, 0, 0, 1]]
    return Ry

def rotZ(theta):
    Rz = [[np.cos(theta),-np.sin(theta) , 0,0], [np.sin(theta) , np.cos(theta), 0,0], [0, 0, 1,0], [0, 0, 0, 1]]
    return Rz

def save_which_PCD_was_saved(value, tree, imageNumber):
    fields = ['imageName', 'imageNumber', 'typeOfPCD']
    data = [tree, imageNumber,value]
    if(os.path.exists("popisOblakaKMeansRansac.csv")):
        print("wrote")
        with open("popisOblakaKMeansRansac.csv", 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
    else:
        print("wrote_prvi put")
        with open("popisOblakaKMeansRansac.csv", 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(data )

def save_which_PCD_was_savedA(value, tree, imageNumber):
    fields = ['imageName', 'imageNumber', 'typeOfPCD']
    data = [tree, imageNumber,value]
    if(os.path.exists("popisOblakaKMeansRansac_After.csv")):
        print("wrote  after")
        with open("popisOblakaKMeansRansac_After.csv", 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
    else:
        print("wrote_prvi put u After")
        with open("popisOblakaKMeansRansac_After.csv", 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(data )

def getStoredValueFromBefore(tree, imageTurn):
    treeNumber = tree.split("_")[-1]
    listOfUsedNumbers = []
    fileName = "popisOblakaKMeansRansac" + ".csv"
    dataFrame = pd.read_csv(fileName)
    if(After):
        return True
    try:
        methodUsedForRemovingGrass = dataFrame[(dataFrame.imageName == tree)  & (dataFrame.imageNumber == imageTurn)].typeOfPCD.values[0]
    except:
        return True
    return methodUsedForRemovingGrass

def getKoristeneSlike(tree,After):
    treeNumber = tree.split("_")[-1]
    print(treeNumber)
    listOfUsedNumbers = []
    if not After:
        fileName = "Indexes_Before_Pruning_BranchVol2" + ".csv"
        dataFrame = pd.read_csv(fileName)
        strOfUsedNumbers = dataFrame.iloc[int(treeNumber),2]
        print(fileName)
    else:
        fileName = "Indexes_After_Pruning_BranchVol2" + ".csv"
        dataFrame = pd.read_csv(fileName)
        strOfUsedNumbers = dataFrame.iloc[int(treeNumber),2]
    return strOfUsedNumbers

def execute_teaser_global_registration(source_points, target_points):
    """
    Perform global registration using TEASER++
    """
    # Initialize TEASER++ solver
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = NOISE_BOUND
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12
    
    teaser_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    # Solve
    start = timer()
    teaser_solver.solve(source_points, target_points)
    end = timer()
    
    # Extract results
    solution = teaser_solver.getSolution()
    transformation = np.eye(4)
    transformation[:3, :3] = solution.rotation
    transformation[:3, 3] = solution.translation
    
    print(f"TEASER++ registration completed in {end - start:.4f} seconds.")
    return transformation, end-start



def find_correspondences(source_fpfh, target_fpfh):
    """
    Find correspondences using KDTreeFlann search.
    """
    source_features = np.asarray(source_fpfh.data).T
    target_features = np.asarray(target_fpfh.data).T

    print("Finding correspondences...")
    tree = o3d.geometry.KDTreeFlann()
    tree.set_matrix_data(target_features.T)  # Build KDTree on target features

    source_corrs = []
    target_corrs = []

    for i, feature in enumerate(source_features):
        
        [_, idx, _] = tree.search_knn_vector_xd(feature, 1)  # Find nearest neighbor
        if idx:  # Ensure a match is found
            source_corrs.append(i)  # Index in source
            target_corrs.append(idx[0])  # Index in target
    
    return np.array(source_corrs), np.array(target_corrs)



def calculate_IoU (source_temp,target_temp):
    #IntersectionOverUnion
    numOfSourcePoints = len(source_temp.points)
    numOfTargetPoints = len(target_temp.points)
    dists = np.array(source_temp.compute_point_cloud_distance(target_temp))
    threshold_1cm = 0.01 #1 cm threshold
    threshold_5cm = 0.05 #5 cm threshold
    ind_1cm = np.where(dists > threshold_1cm)[0]
    ind_5cm = np.where(dists > threshold_5cm)[0]
    #Select by index (remove all that satistfy the condition above)
    outliers_1cm = source_temp.select_by_index(ind_1cm)
    outliers_5cm = source_temp.select_by_index(ind_5cm)
    correspondencePoints_1cm = source_temp.select_by_index(ind_1cm, invert=True)
    correspondencePoints_5cm = source_temp.select_by_index(ind_5cm, invert=True)
    numOfCorrespondencePoints_1cm = numOfSourcePoints - len(outliers_1cm.points)
    numOfCorrespondencePoints_5cm = numOfSourcePoints - len(outliers_5cm.points)
    numOfUnionPoints_1cm = numOfSourcePoints + numOfTargetPoints - numOfCorrespondencePoints_1cm
    numOfUnionPoints_5cm = numOfSourcePoints + numOfTargetPoints - numOfCorrespondencePoints_5cm
    print("Number of points in source (reconstruced PCD): " + str(numOfSourcePoints))
    print("Number of points in target (synthetic model): " + str(numOfTargetPoints))
    print("Number of correspondence points for threshold 1cm (intersection): " + str(numOfCorrespondencePoints_1cm))
    print("Number of correspondence points for threshold 5cm (intersection): " + str(numOfCorrespondencePoints_5cm))
    print("Number of outliers for threshold 1cm: " + str(len(outliers_1cm.points)))
    print("Number of outliers for threshold 5cm: " + str(len(outliers_5cm.points)))
    print("Number of points of both point clouds for threshold 1cm (union): " + str(numOfUnionPoints_1cm))
    print("Number of points of both point clouds for threshold 5cm (union): " + str(numOfUnionPoints_5cm))
    print("Intersection over union for threshold 1cm: " + str(numOfCorrespondencePoints_1cm / (numOfUnionPoints_1cm)))
    print("Intersection over union for threshold 5cm: " + str(numOfCorrespondencePoints_5cm / (numOfUnionPoints_5cm)))
    return numOfCorrespondencePoints_1cm / (numOfUnionPoints_1cm) , numOfCorrespondencePoints_5cm / (numOfUnionPoints_5cm)


def save_Teaser_ICP_values(voxel_size, B_A, tree, string_ofUsedImages, teaser_fitness, teaser_inlier_rmse, teaser_corrSetSize, teaser_transformation, teaser_time, icp_fitness, icp_inlier_rmse, icp_corrSetSize, icp_transformation, icp_time, voxel002 ):
    """
    To CSV file save next data:
        -- tree -> which tree we are working on
        -- string_ofUsedImages -> used images in that tree
        -- teaser_fitness -> fitness value of corresponding set, greater is better
        -- teaser_inlier_rmse -> inlier rmse of corresponding set, smaller is better
        -- teaser_corrSetSize -> size of corresponding data, 
        -- teaser_time -> time needed for teaser registration
    """
    
    destination_path = "/home/user/BRANCH_v2/images/asus/" + B_A +"/E/" 
    # Define the file path for CSV
    if(voxel002 and B_A == "B"):
        csv_file_path = os.path.join(destination_path, "AboutTEASER_and_ICP__perTree_Voxel002_Before.csv") 
    if(voxel002==False and B_A == "B"):
        csv_file_path = os.path.join(destination_path, "AboutTEASER_and_ICP__perTree_Before.csv") 
    if(voxel002 and B_A == "A"):
        csv_file_path = os.path.join(destination_path, "AboutTEASER_and_ICP__perTree_Voxel002_After.csv") 
    if(voxel002==False and B_A == "A"): 
        csv_file_path = os.path.join(destination_path, "AboutTEASER_and_ICP__perTree_After.csv") 
    # print("file_ ", destination_path)
    # print("stop")
    # Ensure the destination directory exists
    os.makedirs(destination_path, exist_ok=True)

    fields = ['voxel_size', 'tree', 'string_ofUsedImages', 'teaser_fitness', 'teaser_inlier_rmse', 'teaser_corrSetSize', 'teaser_transformation','teaser_time', 'icp_fitness', 'icp_inlier_rmse', 'icp_corrSetSize', 'icp_transformation','icp_time']
    data = [voxel_size, tree, string_ofUsedImages, teaser_fitness, teaser_inlier_rmse, teaser_corrSetSize, teaser_transformation, teaser_time, icp_fitness, icp_inlier_rmse, icp_corrSetSize, icp_transformation, icp_time ]
    if(os.path.exists(csv_file_path)):
        #print("wrote")
        with open(csv_file_path, 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(data)
    else:
        #print("wrote_first")
        with open(csv_file_path, 'a',newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(fields)
            csvwriter.writerow(data )



def save_voxelized_pcd (destination_path, pcd_voxeled, k, voxel002):
    if(voxel002):
        destination_path = path_to_tree_folder + "/reconstruction_Voxel002/voxelized_pc/" 
    else:
        destination_path = path_to_tree_folder + "/reconstruction/voxelized_pc/" 
    os.makedirs(destination_path, exist_ok=True)

    full_path = destination_path + str(k)+ ".ply"
    o3d.io.write_point_cloud(full_path, pcd_voxeled)


def remove_createdDirs_Before_Evening(start, top,voxel002, B_A):
    base_path = "/home/user/BRANCH_v2/images/asus/" + B_A +"/E/"
    print("----Check before deleting ",base_path)

    # Iterate over the tree directories
    for i in range(start,top):  # Adjust the range as needed (e.g., 100 for `tree_1_v_0000` to `tree_1_v_0099`)
        if(voxel002):
            tree_dir = os.path.join(base_path, f"tree_1_V_{i:04d}/reconstruction_Voxel002" )  
        else:
            tree_dir = os.path.join(base_path, f"tree_1_V_{i:04d}/reconstruction")  
        
        # Check if the directory exists
        if os.path.exists(tree_dir):
            # Remove the Filtered__noGrass directory
            shutil.rmtree(tree_dir)
            print(f"Removed: {tree_dir}")
        else:
            print(f"Directory not found: {tree_dir}")

#REMOVE 
#remove_createdDirs_Before_Evening(20,21,voxel002, "B")



#In all files there is "_Voxel002" as addition

#TEASER and ICP
voxel_size = 0.02 #0.01   #1cm and 2cm
# TEASER++ Parameters
NOISE_BOUND = voxel_size #should be same as voxel size
visualisationFlag = False
After = False
starting_number = 9
imageNumber = 0
row_number = 1
sorta = "V"
if After:
    B_A = "A"
else:
    B_A = "B"
do_ICP = False
saveData = False
saveTimeData = False

if(voxel_size == 0.02):
    voxel002 = True
else:
    voxel002 = False



print("voxel_size is 0.02: ", voxel002)
for i in range(starting_number,184): 
    imageNumber = 0

    if(i<10):
        tree="tree_"+ str(row_number)+ "_" + sorta + "_" + "000" + str(i)
    elif(i>=10 and i <= 99):
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "00" + str(i)
    else:
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "0" + str(i)

    #SST 
    path_to_tree_folder = "/home/user/BRANCH_v2/images/asus/" + B_A +"/E/" + tree 
    full_path = "/home/user/BRANCH_v2/images/asus/" + B_A+ "/E/" + tree + "/Filtered__noGrass" #Filtered_noGrass"   #angle0"

    #print("path_to_tree_folder")
    lst = os.listdir(path_to_tree_folder + "/angle0/color") 
    numberOfImagesOfOneTree = len(lst)
    print(numberOfImagesOfOneTree)
    all_PCD_One_Tree = []

    string_ofUsedImages = getKoristeneSlike(tree,After)
    print("Pročitane točke su sljedeće: ", string_ofUsedImages)
    imageTurn = 0
    if(string_ofUsedImages == "-"):
        continue
    

    for im in range(0, numberOfImagesOfOneTree+1):
        if(im in np.array(string_ofUsedImages.split(","), dtype = int)):
            
            print("Working on image: " + tree + " , number: "+ str(im))
            # #print(full_path + "/color/" +str(im) + ".png")
            # color_bp = o3d.io.read_image(full_path + "/color/" +str(im) + ".png")
            # depth_bp = o3d.io.read_image(full_path+"/depth/" + str(im) + ".png")
 

            # rgbd_bp = o3d.geometry.RGBDImage.create_from_color_and_depth(color_bp, depth_bp, convert_rgb_to_intensity=False)
            # pcd_bp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_bp, camera_depth)
            
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

            # pcd_bp.transform(rotZ(-3.14/2))# #Rz -90
            # pcd_bp.transform(rotY(3.14))# #RY 180
            print("path: ", full_path+"/pc/" + str(im) + ".ply")
            pcd_bp = o3d.io.read_point_cloud(full_path+"/pc/" + str(im) + ".ply")
            if(visualisationFlag):
                o3d.visualization.draw_geometries([pcd_bp], window_name = "ORIGINAL RGB-D: "+ str(tree))
                o3d.visualization.draw_geometries([pcd_bp], window_name = "ORIGINAL RGB-D: "+ str(tree))
            all_PCD_One_Tree.append(pcd_bp)




    #reconstruction of a 3D tree model based on 4 pictures taken
    #target is fixed, source rotates

    list_TeaserTime = []
    target = all_PCD_One_Tree[0]
    variable_forTime = 0.0
    list_IcpTime = []
    best_rmse = 999.9
    threshold = voxel_size

    values__teaser_fitness = []
    values__icp_fitness = []

    values__teaser_inlier_rmse = []
    values__icp_inlier_rmse = []

    values__teaser_corr_set_size = []
    values__icp_corr_set_size = []

    transformation_teaser_list = []
    transformation_icp_list = []

    voxelising_time_list = []
    correspondence_time_list = []
    teaser_time_list = []
    sum_TeaserTime_list = []

    
    # # #TEASER++ and ICP start
    for k in range(0, 4-1):

        if(k == 0):
            target_teaser = target
            target_icp = target
            

        source = all_PCD_One_Tree[k+1]
        source_colors = np.asarray(source.colors) 
        target_colors_teaser = np.asarray(target_teaser.colors) 
        target_colors_icp = np.asarray(target_icp.colors) 
  

        # Downsample point clouds and Extract features
        start_voxeling_fpfh = time.time()
        source_pcd, source_fpfh = preprocess_point_cloud(source, voxel_size)
        target_pcd_teaser, target_fpfh_teaser = preprocess_point_cloud(target_teaser, voxel_size)
        target_pcd_icp, target_fpfh_icp = preprocess_point_cloud(target_icp, voxel_size)
        stop_voxeling_fpfh  = time.time()     

        if(k == 0):
            if saveData:
                save_voxelized_pcd(path_to_tree_folder, target_pcd_teaser, k,voxel002)
        if saveData:
            save_voxelized_pcd(path_to_tree_folder, source_pcd, k+1,voxel002)

        if visualisationFlag:
            o3d.visualization.draw_geometries([source_pcd, target_pcd_teaser], "Downsampled preprocessed point cloud - Teaser")  
            o3d.visualization.draw_geometries([source_pcd, target_pcd_icp], "Downsampled preprocessed point cloud - ICP ")  
        

        # Find correspondences for TEASER
        start_correspondences = time.time()
        source_corr_teaser, target_corr_teaser = find_correspondences(source_fpfh, target_fpfh_teaser)#, source_pcd, target_pcd)
        stop_correspondences = time.time()


        # Extract the corresponding points using the indices for TEASER
        source_corr_teaser = np.asarray(source_pcd.points)[source_corr_teaser].T            # Shape (3, N)
        target_corr_teaser = np.asarray(target_pcd_teaser.points)[target_corr_teaser].T     # Shape (3, N)
        

        # Perform TEASER++ Registration
        transformation_teaser, teaser_time = execute_teaser_global_registration(source_corr_teaser, target_corr_teaser)
        
        
        # evaluation_result_teaser = o3d.pipelines.registration.evaluate_registration(
        #     source, target_teaser, threshold, transformation_teaser)

        evaluation_result_teaser_voxel = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd_teaser, threshold, transformation_teaser)
        

        values__teaser_fitness.append(evaluation_result_teaser_voxel.fitness)
        values__teaser_inlier_rmse.append(evaluation_result_teaser_voxel.inlier_rmse)
        values__teaser_corr_set_size.append(len(evaluation_result_teaser_voxel.correspondence_set))
        transformation_teaser_list.append(transformation_teaser)
        
        

        # # Visualize the result
        if visualisationFlag:
            draw_registration_result(source_pcd, target_pcd_teaser, transformation_teaser, tree, imageNumber, "TEASER transformation number: "+ str(k))

        #[source_temp_teaser, target_temp_teaser] = draw_registration_result(source, target_teaser, transformation_teaser, tree, imageNumber, "TEASER number: "+ str(k))

        source_temp_teaser = copy.deepcopy(source)
        target_temp_teaser = copy.deepcopy(target_teaser)
        source_temp_teaser.paint_uniform_color([1, 0, 0])
        target_temp_teaser.paint_uniform_color([0, 0, 1])
        source_temp_teaser.transform(transformation_teaser)
        #o3d.visualization.draw_geometries([target_temp_teaser + source_temp_teaser], window_name = "red_blue TARGET : "+ str(tree))
        
        sum_TeaserTime = (stop_voxeling_fpfh - start_voxeling_fpfh) + \
                        (stop_correspondences - start_correspondences) + teaser_time
        print("Time nedded for Teaser altogether: ", teaser_time , " sec")
        print("Time nedded for Teaser: ", sum_TeaserTime , " sec")
        
        list_TeaserTime.append(teaser_time)
        target_temp_teaser.colors = o3d.utility.Vector3dVector(target_colors_teaser)
        source_temp_teaser.colors = o3d.utility.Vector3dVector(source_colors)
        target_teaser = target_temp_teaser + source_temp_teaser
        o3d.visualization.draw_geometries([target_teaser], window_name = "teaser TARGET : "+ str(tree))

        voxelising_time_list.append(stop_voxeling_fpfh - start_voxeling_fpfh)
        correspondence_time_list.append(stop_correspondences - start_correspondences)
        teaser_time_list.append(teaser_time)
        sum_TeaserTime_list.append(sum_TeaserTime)

        # Perform ICP 
        if (do_ICP):
            z = 0
            stopper = 0
            best_correspondence_ransac = 0
            best_source_temp_color_icp = np.asarray(source.colors) 
            best_fake = 0
            best_transformation = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

            #just repetition to see if in 5 attempts he can get a great solution
            while(z < 5):
            
                [result_ransac, ransac_time] = execute_global_registration(source_pcd, target_pcd_icp, source_fpfh, target_fpfh_icp,voxel_size)
    
                #print(result_ransac)

                if visualisationFlag:
                    draw_registration_result(source_pcd, target_pcd_icp, result_ransac.transformation, tree, imageNumber, "Ransac transformation number: "+ str(k))

                #print("Initial alignment")
                evaluation = o3d.pipelines.registration.evaluate_registration(source_pcd, target_pcd_icp, threshold, result_ransac.transformation)
                #print(evaluation)

        
                result_icp, time_icp = refine_registration(source_pcd, target_pcd_icp, source_fpfh, target_fpfh_icp, voxel_size, result_ransac)
                print("----Result after ICP, pokusaj: ", z)
                print(result_icp)    

            

                #[source_temp_icp, target_temp_icp]= draw_registration_result(source, target_icp, result_icp.transformation, tree, imageNumber, "ICP transformation" + str(k))
                
                source_temp_icp = copy.deepcopy(source)
                target_temp_icp = copy.deepcopy(target_icp)
                source_temp_icp.paint_uniform_color([1, 0, 0])
                target_temp_icp.paint_uniform_color([0, 0, 1])
                source_temp_icp.transform(result_icp.transformation)
                
                #o3d.visualization.draw_geometries([all_PCD_One_Tree[k]], window_name = "Colored image before pruning")
                #o3d.visualization.draw_geometries([all_PCD_One_Tree[k+1]], window_name = "Colored image after pruning")
                if len(result_icp.correspondence_set) > best_correspondence_ransac: 
                    if((z > 0) & (result_icp.inlier_rmse > best_fake*1.2)):
                        print("OVO JE SAD TAJ PROBLEM 3 GDJE IMA VIŠE TOČAKA ALI MANJU PODUDARNOST")
                    else:
                        best_source_temp_icp = source_temp_icp
                        best_source_temp_color_icp = np.asarray(source.colors)
                        best_correspondence_ransac = len(result_icp.correspondence_set)
                        if(k <1 ): best_rmse = result_icp.inlier_rmse #remember the best rmse for overlapping the first 2 images  (this is taken as reference)
                        best_fake = result_icp.inlier_rmse
                        print("so far najbolji je: ", str(z))
                        print("- --- --- --- --- --- --- --- -- --- --- -- --- -- --- -- -- -- -- --- -- -- --- --- -- --- -- --- --- -- --- --- -- --- --- -- --- -- --- --- -- -")
                        variable_forTime = time_icp
                        best_transformation = result_icp.transformation

                
                z = z+1
            #za spajanje skupa od 3 slike s 4-tom slikom aktivira ovaj slučaj (jer prije svakako nadje najbolje rj),
            #ideja je da se vrti dokle god ne nadje 50% losiji rezultat od najboljeg pronadjenog rezultata za spajanje
            #prve dvije slike jer tu pronalazi najbolji rmse od svih spajanja (plus rmse se povecava sa svakom novom dodanom slikom zato i mora biti raspon od 50% )
            ## TRANSLATION: #to merge a set of 3 images with the 4th image activates this case (because it certainly finds the best solution before),
            #the idea is to loop until it finds a result 50% worse than the best result found for the merge
            #the first two images because that's where he finds the best rmse of all connections (plus the rmse increases with each new added image, 
            # that's why it has to be a range of 50%)
                if(z > 4 and best_rmse*1.5 < best_fake):
                    z = z-1
                    stopper = stopper+1
                    print(" ! ! ! ! ! ! Repeating while there was no best overlap found")
                    if(stopper > 10):
                        break

            if (stopper < 9):
                evaluation_result_icp_voxel = o3d.pipelines.registration.evaluate_registration(
                    source_pcd, target_pcd_icp, threshold, best_transformation)
                # evaluation_result_icp = o3d.pipelines.registration.evaluate_registration(
                #     source, target_icp, threshold, best_transformation)
                
                values__icp_fitness.append(evaluation_result_icp_voxel.fitness)
                values__icp_inlier_rmse.append(evaluation_result_icp_voxel.inlier_rmse)
                values__icp_corr_set_size.append(len(evaluation_result_icp_voxel.correspondence_set))

                best_source_temp_icp.colors =  o3d.utility.Vector3dVector(best_source_temp_color_icp)
                target_temp_icp.colors = o3d.utility.Vector3dVector(target_colors_icp)
                target_icp = target_temp_icp + best_source_temp_icp
                #o3d.visualization.draw_geometries([target_icp], window_name = "check color for ICP result: "+ str(tree))

                list_IcpTime.append(variable_forTime)
                transformation_icp_list.append(best_transformation)


    print("final results: \n")
    #o3d.visualization.draw_geometries([target_teaser, origin_frame], window_name = "TEASER DONE FOR: "+ str(tree))
    #o3d.visualization.draw_geometries([target_icp, origin_frame], window_name = "ICP DONE FOR: "+ str(tree))

    if(do_ICP):
        if saveData:
            save_Teaser_ICP_values(voxel_size, B_A, tree, string_ofUsedImages, values__teaser_fitness, values__teaser_inlier_rmse, values__teaser_corr_set_size, transformation_teaser_list, list_TeaserTime, values__icp_fitness, values__icp_inlier_rmse, values__icp_corr_set_size, transformation_icp_list, list_IcpTime,voxel002)
    if saveData:
        save_recontructed_PCD(target_teaser, "teaser",tree, path_to_tree_folder,voxel002)
        save_recontructed_PCD(target_icp, "icp",tree, path_to_tree_folder,voxel002)
    print(" +++++++++++++++++++++++++++++++++++++++++++++++ DONE")

    if saveTimeData:
        save_timeNeededForTeaser(tree, voxelising_time_list,correspondence_time_list,teaser_time_list,sum_TeaserTime_list)




    