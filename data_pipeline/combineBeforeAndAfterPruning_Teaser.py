#Preklop PCD-a dvije slike 
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
import time
import teaserpp_python 
from timeit import default_timer as timer

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
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)) #search parameters for hybrid knn and radial search

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
    

def draw_registration_result(source, target, transformation, tree, window_name):
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

def save_recontructed_PCD(merged_pcd, treeData,full_path):
    name = treeData + "_mergedBA.ply"
    o3d.io.write_point_cloud(full_path + "\\" + name, merged_pcd)
    print("Saved merged point cloud")


def rotX(theta):
    Rx = [[1, 0 , 0,0], [0 , np.cos(theta), -np.sin(theta),0], [0, np.sin(theta), np.cos(theta),0],[0, 0, 0, 1]]
    return Rx

def rotY(theta):
    Ry = [[np.cos(theta), 0 , np.sin(theta),0], [0 , 1, 0,0], [ -np.sin(theta),0, np.cos(theta),0],[0, 0, 0, 1]]
    return Ry

def rotZ(theta):
    Rz = [[np.cos(theta),-np.sin(theta) , 0,0], [np.sin(theta) , np.cos(theta), 0,0], [0, 0, 1,0], [0, 0, 0, 1]]
    return Rz


def save_voxelized_pcd (destination_path, pcd_voxeled, k, voxel002):
    if(voxel002):
        destination_path = destination_path + "/voxel_002/" 
    else:
        destination_path = destination_path + "/voxel_001/" 
    os.makedirs(destination_path, exist_ok=True)

    full_path = destination_path + str(k)+ ".ply"
    print("saved to: ", full_path)
    o3d.io.write_point_cloud(full_path, pcd_voxeled)


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
        # if i % 1000 == 0:
        #     print('feature %d' % i)
        [_, idx, _] = tree.search_knn_vector_xd(feature, 1)  # Find nearest neighbor
        if idx:  # Ensure a match is found
            source_corrs.append(i)  # Index in source
            target_corrs.append(idx[0])  # Index in target
    
    return np.array(source_corrs), np.array(target_corrs)

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


def colorDistinctiveBranches(target_temp, best_source_temp, voxel_size):

    distane_threshold = 0.03
    start_timer_labels = time.time()
    dists = target_temp.compute_point_cloud_distance(best_source_temp)
    dists = np.asarray(dists)
    ind = np.where(dists > float(distane_threshold))[0]

    best_source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])

    branches_colors = np.array(target_temp.colors)
    branches_colors[ind] = [1.0, 1.0, 0.0] #yellow

    target_temp.colors = o3d.utility.Vector3dVector(branches_colors)
    target_2 = target_temp + best_source_temp
    print("Prikaz grana žutom bojom")
    #o3d.visualization.draw_geometries([target_2], window_name = "Stablo: "+ str(tree))
    stop_timer_labels = time.time()
    return target_2, stop_timer_labels - start_timer_labels



def save_distinctive_recontructed_PCD(merged_pcd, teaser_or_icp, treeData,full_path, voxel002):
    if(voxel002):
        destination_path = full_path + "/distinctive_merged/voxel_002_mobdok/"
    else:
        destination_path = full_path + "/distinctive_merged/voxel_001/"
    #destination_path = full_path + "/distinctive_merged/"
    os.makedirs(destination_path, exist_ok=True)
    name = treeData +"_" + teaser_or_icp+".ply"
    o3d.io.write_point_cloud(destination_path + "/" + name, merged_pcd)
    print("Saved merged point cloud")


def save_ICP_values(full_path_merged, voxel_size, tree, \
            values__icp_fitness,    values__icp_inlier_rmse,    values__icp_corr_set_size,    transformation_icp,    variable_forTime_icp, \
            voxel002):
    """
    To CSV file save next data:
        -- tree -> which tree we are working on
        -- string_ofUsedImages -> used images in that tree
        -- teaser_fitness -> fitness value of corresponding set, greater is better
        -- teaser_inlier_rmse -> inlier rmse of corresponding set, smaller is better
        -- teaser_corrSetSize -> size of corresponding data, 
        -- teaser_time -> time needed for teaser registration
    """
    
    # Define the file path for CSV
    if(voxel002):
        csv_file_path = os.path.join(full_path_merged + "/distinctive_merged/voxel_002_mobdok/","AboutICP__perTree_Voxel002.csv") 
        make_dirs = full_path_merged + "/distinctive_merged/voxel_002_mobdok"
    else:
        csv_file_path = os.path.join(full_path_merged + "/distinctive_merged/voxel_001", "AboutICP__perTree_Voxel001.csv") 
        make_dirs = full_path_merged + "/distinctive_merged/voxel_001"

    
    os.makedirs(make_dirs, exist_ok=True)

    fields = ['voxel_size', 'tree', 
              'icp_fitness',    'icp_inlier_rmse',    'icp_corrSetSize',    'icp_transformation',    'icp_time']
    data = [voxel_size, tree, 
            values__icp_fitness,    values__icp_inlier_rmse,    values__icp_corr_set_size,    transformation_icp,    variable_forTime_icp]
    
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



def save_Teaser_values(full_path_merged, voxel_size, tree, \
            values__teaser_fitness, values__teaser_inlier_rmse, values__teaser_corr_set_size, transformation_teaser, teaser_time, \
          voxel002):

    # Define the file path for CSV
    if(voxel002):
        csv_file_path = os.path.join(full_path_merged + "/distinctive_merged/voxel_002_mobdok/","AboutTEASER__perTree_Voxel002.csv") 
        make_dirs = full_path_merged + "/distinctive_merged/voxel_002_mobdok"
    else:
        csv_file_path = os.path.join(full_path_merged + "/distinctive_merged/voxel_001", "AboutTEASER__perTree_Voxel001.csv") 
        make_dirs = full_path_merged + "/distinctive_merged/voxel_001"

    os.makedirs(make_dirs, exist_ok=True)

    fields = ['voxel_size', 'tree', 
              'teaser_fitness', 'teaser_inlier_rmse', 'teaser_corrSetSize', 'teaser_transformation','teaser_time']
    data = [voxel_size, tree, 
            values__teaser_fitness, values__teaser_inlier_rmse, values__teaser_corr_set_size, transformation_teaser, teaser_time  ]
    
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

def save_timeNeededForTeaser(tree, voxelising_time_list,correspondence_time_list,teaser_time_list,sum_TeaserTime_list,time_labeling):
    destination_path = "/home/user/BRANCH_v2/images/asus/Merged" + "/distinctive_merged/voxel_002_mobdok/"
    csv_file_path = os.path.join(destination_path, "MDPI_times_BA.csv") 
    fields = ['voxel_size', 'tree', 'voxelisingANDfpfh_time', 'correspondence_time', 'teaser_time', 'sum_TeaserTime', 'labeling_time']
    data = [voxel_size, tree,  voxelising_time_list, correspondence_time_list, teaser_time_list, sum_TeaserTime_list,  time_labeling]
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



visualisationFlag = False
starting_number = 0
row_number = 1
sorta = "V"
saveResults = False
voxel_size = 0.02 #1cm or 0.02
NOISE_BOUND = voxel_size #should be same as voxel size
saveTimeData = True
skipICP = True

for i in range(starting_number,184): 

    if(i<10):
        tree="tree_"+ str(row_number)+ "_" + sorta + "_" + "000" + str(i)
    elif(i>=10 and i <= 99):
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "00" + str(i)
    else:
        tree="tree_" + str(row_number)+ "_" + sorta + "_" + "0" + str(i)

    #SST 
    full_path_Before = "/home/user/BRANCH_v2/images/asus/B/E/" + tree + "/reconstruction_Voxel002"
    full_path_After = "/home/user/BRANCH_v2/images/asus/A/E/" + tree + "/reconstruction_Voxel002"
    full_path_merged = "/home/user/BRANCH_v2/images/asus/Merged"

            
    print("Working on image: " + tree)    
    pcd_before = o3d.io.read_point_cloud(full_path_Before + "/" + tree + "_merged_teaser.ply")
    if(visualisationFlag):
        o3d.visualization.draw_geometries([pcd_before], window_name = "ORIGINAL RGB-D: "+ str(tree))

    pcd_after = o3d.io.read_point_cloud(full_path_After + "/" + tree + "_merged_teaser.ply")
    if not pcd_after.has_points() or not pcd_before.has_points():
        print("No data loaded. Not working on tree: ", tree)
        continue

    if(visualisationFlag):
        o3d.visualization.draw_geometries([pcd_after], window_name = "ORIGINAL RGB-D: "+ str(tree))

    #target is fixed, source is being rotated
    target = pcd_before
    best_rmse = 999.9

    if(voxel_size == 0.02):
        voxel002 = True
    else:
        voxel002 = False

    source = pcd_after
    source_colors = np.asarray(source.colors) 
    target_colors = np.asarray(target.colors) 


    threshold = 0.2
    start_voxeling_fpfh = time.time()
    source_pcd, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_pcd, target_fpfh = preprocess_point_cloud(target, voxel_size)
    stop_voxeling_fpfh  = time.time()   
    path_to_tree_folder = "/home/user/BRANCH_v2/images/asus/Merged/voxelized_pcd/" + tree

    if(saveResults):
        save_voxelized_pcd(path_to_tree_folder, source_pcd, 0,voxel002)
        save_voxelized_pcd(path_to_tree_folder, target_pcd, 1,voxel002)

    if visualisationFlag:
        o3d.visualization.draw_geometries([source_pcd,target_pcd], "Downsampled preprocessed point cloud")
    
    # Find correspondences for TEASER
    start_correspondences = time.time()
    source_corr_teaser, target_corr_teaser = find_correspondences(source_fpfh, target_fpfh)#, source_pcd, target_pcd)
    stop_correspondences = time.time()


    # Extract the corresponding points using the indices for TEASER
    source_corr_teaser = np.asarray(source_pcd.points)[source_corr_teaser].T     # Shape (3, N)
    target_corr_teaser = np.asarray(target_pcd.points)[target_corr_teaser].T     # Shape (3, N)
        
    # Perform TEASER++ Registration
    transformation_teaser, teaser_time = execute_teaser_global_registration(source_corr_teaser, target_corr_teaser)
        
    evaluation_result_teaser_voxel = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd, threshold, transformation_teaser)
        
    if visualisationFlag:
            draw_registration_result(source_pcd, target_pcd, transformation_teaser, tree, "TEASER transformation")

    values__teaser_fitness = evaluation_result_teaser_voxel.fitness
    values__teaser_inlier_rmse = evaluation_result_teaser_voxel.inlier_rmse
    values__teaser_corr_set_size = len(evaluation_result_teaser_voxel.correspondence_set)

    source_temp_teaser = copy.deepcopy(source)
    target_temp_teaser = copy.deepcopy(target)
    source_temp_teaser.paint_uniform_color([1, 0, 0])
    target_temp_teaser.paint_uniform_color([0, 0, 1])
    source_temp_teaser.transform(transformation_teaser)
    
    sum_TeaserTime = (stop_voxeling_fpfh - start_voxeling_fpfh) + \
                    (stop_correspondences - start_correspondences) + teaser_time
    print("Time nedded for Teaser: ", teaser_time , " sec")
    print("Time nedded for Teaser altogether: ", sum_TeaserTime , " sec")
    
    if visualisationFlag:
        o3d.visualization.draw_geometries([target_temp_teaser,source_temp_teaser], window_name = "TEASER RESULT: "+ str(tree))

    target_temp_teaser.colors = o3d.utility.Vector3dVector(target_colors)
    source_temp_teaser.colors = o3d.utility.Vector3dVector(source_colors)
    target_teaser = target_temp_teaser + source_temp_teaser
    if visualisationFlag:
        o3d.visualization.draw_geometries([target_teaser], window_name = "TEASER RESULT: "+ str(tree))

    merged_distinctive_pcd_teaser,time_labeling = colorDistinctiveBranches(target_temp_teaser, source_temp_teaser, 0.01)
    if(saveResults):
        save_distinctive_recontructed_PCD(merged_distinctive_pcd_teaser, "teaser", tree,full_path_merged, voxel002)

        save_Teaser_values(full_path_merged, voxel_size, tree, \
        values__teaser_fitness, values__teaser_inlier_rmse, values__teaser_corr_set_size, transformation_teaser, teaser_time, \
     voxel002)
    if saveTimeData:
        save_timeNeededForTeaser(tree, stop_voxeling_fpfh - start_voxeling_fpfh,stop_correspondences - start_correspondences,teaser_time,sum_TeaserTime, time_labeling)


    if not  skipICP:
        # Perform ICP 
        z = 0
        stopper = 0
        best_correspondence_ransac = 0
        best_source_temp_color_icp = np.asarray(source.colors) 
        best_fake = 0
        best_transformation = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])

        #samo ponavljanje da vidim jel moze u 5 pokusaja doci do nekog supac rj
        while(z < 5):
            
            [result_ransac, ransac_time] = execute_global_registration(source_pcd, target_pcd, source_fpfh, target_fpfh,voxel_size)
            #print(result_ransac)

            if visualisationFlag:
                draw_registration_result(source_pcd, target_pcd, result_ransac.transformation, tree, "Ransac transformation")

            #print("Initial alignment")
            evaluation = o3d.pipelines.registration.evaluate_registration(source_pcd, target_pcd, threshold, result_ransac.transformation)
            #print(evaluation)

        
            result_icp, time_icp = refine_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, voxel_size, result_ransac)
            print("----Result after ICP, pokusaj: ", z)
            print(result_icp)    


            source_temp_icp = copy.deepcopy(source)
            target_temp_icp = copy.deepcopy(target)
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
                    best_fake = result_icp.inlier_rmse
                    print("so far najbolji je: ", str(z))
                    print("- --- --- --- --- --- --- --- -- --- --- -- --- -- --- -- -- -- -- --- -- -- --- --- -- --- -- --- --- -- --- --- -- --- --- -- --- -- --- --- -- -")
                    variable_forTime_icp = time_icp
                    best_transformation = result_icp.transformation

            
            z = z+1
            #za spajanje skupa od 3 slike s 4-tom slikom aktivira ovaj slučaj (jer prije svakako nadje najbolje rj),
            #ideja je da se vrti dokle god ne nadje 50% losiji rezultat od najboljeg pronadjenog rezultata za spajanje
            #prve dvije slike jer tu pronalazi najbolji rmse od svih spajanja (plus rmse se povecava sa svakom novom dodanom slikom zato i mora biti raspon od 50% )
            ## TRANSLATION: #to merge a set of 3 images with the 4th image activates this case (because it certainly finds the best solution before),
            #the idea is to loop until it finds a result 50% worse than the best result found for the merge
            #the first two images because that's where he finds the best rmse of all connections (plus the rmse increases with each new added image, 
            # that's why it has to be a range of 50%)
        
        evaluation_result_icp_voxel = o3d.pipelines.registration.evaluate_registration(
            source_pcd, target_pcd, threshold, best_transformation)
        # evaluation_result_icp = o3d.pipelines.registration.evaluate_registration(
        #     source, target_icp, threshold, best_transformation)
        
        values__icp_fitness= evaluation_result_icp_voxel.fitness
        values__icp_inlier_rmse = evaluation_result_icp_voxel.inlier_rmse
        values__icp_corr_set_size = len(evaluation_result_icp_voxel.correspondence_set)

        transformation_icp = best_transformation


        best_source_temp_icp.colors =  o3d.utility.Vector3dVector(best_source_temp_color_icp)
        target_temp_icp.colors = o3d.utility.Vector3dVector(target_colors)
        target_icp = target_temp_icp + best_source_temp_icp
        #o3d.visualization.draw_geometries([target_icp], window_name = "check color for ICP result: "+ str(tree))
        #o3d.visualization.draw_geometries([target_icp], window_name = "PCD DONE FOR: "+ str(tree))


        merged_distinctive_pcd_icp = colorDistinctiveBranches(target_temp_icp, best_source_temp_icp, 0.01)
        save_distinctive_recontructed_PCD(merged_distinctive_pcd_icp, "icp",tree, full_path_merged, voxel002)

        save_ICP_values(full_path_merged, voxel_size, tree, \
                values__icp_fitness,    values__icp_inlier_rmse,    values__icp_corr_set_size,    transformation_icp,    variable_forTime_icp, \
                voxel002)
    
        print(" +++++++ Finished for tree: ", tree)





    # #save source
    # shouldIsave = input("If the created model is fine, press y")
    # if(shouldIsave == "y"):
    #     save_recontructed_PCD(target_2, tree, full_path_merged)
    #     #read saved ply 
    #     print("SAVED  MERGED PCD")
    #     pcd = o3d.io.read_point_cloud(full_path_merged + "\\" + tree + "_mergedBA.ply")
    #     o3d.visualization.draw_geometries([pcd], window_name = "-----OVO sam spremila: ")