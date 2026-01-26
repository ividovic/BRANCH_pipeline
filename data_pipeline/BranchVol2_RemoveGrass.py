## Author: Jana Dukić
## Removing Grass points from Branch_vol2 dataset and saving pointcloud, RGB-D images without grass
## 25.11.2024. Added save, get functions
## 26.11.2024. Fixed bug with RANSAC stopping program if there is no more than 2 remaining_pcd points in a pointcloud (currently solved using 
##             try/exept but one IF will do the trick: if number_of_remaining_points <=2 there break RANSAC) -> min 3 points needed for plane search
## 03.12.2024. Added option save_Data when flag is set in initialization
## 04.12.2024. Added option to remove all points from found grass_planes 
## 04.12.2024. Added option for searching grass neighbors with creating squares around each grass point
## 09.12.2024. -||- now it works 
## 11.12.2024. Fixed height for grass 0.5m 
## 2.1.2025. Added searching for only bottom part of the tree AND condition wheather there is any grass or not (BY Z axis) --- also automatization of removing saved files 


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
import open3d as o3d
import shutil
import time

def rotX(theta):
    Rx = [[1, 0 , 0,0], [0 , np.cos(theta), -np.sin(theta),0], [0, np.sin(theta), np.cos(theta),0],[0, 0, 0, 1]]
    return Rx

def rotY(theta):
    Ry = [[np.cos(theta), 0 , np.sin(theta),0], [0 , 1, 0,0], [ -np.sin(theta),0, np.cos(theta),0],[0, 0, 0, 1]]
    return Ry

def rotZ(theta):
    Rz = [[np.cos(theta),-np.sin(theta) , 0,0], [np.sin(theta) , np.cos(theta), 0,0], [0, 0, 1,0], [0, 0, 0, 1]]
    return Rz

def save_ColorImageFromPointCloud(path_to_tree_folder, imageNumber, color_imageFromPointCloud):
    """
    Saves an image to a specified folder, creating the folder if it does not exist.
    """
    # IF YOU NEED TO REMOVE FOLDER FROM TERMINAL: rm -rf /home/user/BRANCH_v2/images/asus/B/E/tree_1_V_0000/Filtered__noGrass
    # Ensure the folder exists
    destination_path = path_to_tree_folder + "/Filtered__noGrass/color"
    os.makedirs(destination_path, exist_ok=True)
    
    # Construct the full path for the image file
    file_name = str(imageNumber) + ".png"
    file_path = os.path.join(destination_path, file_name)
    
    # Save the image
    cv2.imwrite(file_path, color_imageFromPointCloud)
    #print(f"Image saved to {file_path}")

def save_DepthImageFromPointCloud(path_to_tree_folder, imageNumber, depth_imageFromPointCloud):
    """
    Saves an depth image to a specified folder, creating the folder if it does not exist.
    """
    # Ensure the folder exists
    destination_path = path_to_tree_folder + "/Filtered__noGrass/depth"
    os.makedirs(destination_path, exist_ok=True)
    
    # Construct the full path for the image file
    file_name = str(imageNumber) + ".png"
    file_path = os.path.join(destination_path, file_name)
    
    # Save the image
    cv2.imwrite(file_path, depth_imageFromPointCloud)
    #print(f"Image saved to {file_path}")

def save_PointCloud(path_to_tree_folder, imageNumber,InliersRepresentingTreeWithoutGrass):
    """
    Save point cloud  - in this case without grass
    """
    destination_path = path_to_tree_folder + "/Filtered__noGrass/pc/"
    os.makedirs(destination_path, exist_ok=True)

    full_path = destination_path + str(imageNumber)+ ".ply"
    o3d.io.write_point_cloud(full_path, InliersRepresentingTreeWithoutGrass)
    ##Uncomment if needed testing
    # print(f"PC saved to {full_path}")
    # Check if saving was OK
    # point_cloud = o3d.io.read_point_cloud(full_path)
    # o3d.visualization.draw_geometries([point_cloud], window_name = "SAVED Point Cloud")


def save_PointCloudForTesting(path_to_tree_folder, imageNumber,InliersRepresentingTreeFROMGrass):
    """
    Save point cloud  - in this case original pcd is normally collored and pcd representing saved tree without grass is colored RED
    """
    destination_path = path_to_tree_folder + "/Filtered__noGrass/pc_testing/"
    os.makedirs(destination_path, exist_ok=True)

    full_path = destination_path + str(imageNumber)+ ".ply"
    o3d.io.write_point_cloud(full_path, InliersRepresentingTreeFROMGrass)
    # point_cloud = o3d.io.read_point_cloud(full_path)
    # o3d.visualization.draw_geometries([point_cloud], window_name = "SAVED Point Cloud ---- FOR TESTING")


def save_OriginPointCloudWithPlanesAndSegmentsOfGrass(path_to_tree_folder, imageNumber,whole_pcd, plane_meshes):
    """
    Save point cloud  - in this case with all visualization of original pcd and grass while THE grass plane separately; save all found planes
    """
    destination_path = path_to_tree_folder + "/Filtered__noGrass/pc_accessories/" + str(imageNumber) + "/"
    os.makedirs(destination_path, exist_ok=True)

    combined_points = np.concatenate((np.asarray(whole_pcd[1].points), np.asarray(whole_pcd[0].points)), axis=0)
    combined_colors = np.concatenate((np.asarray(whole_pcd[1].colors), np.asarray(whole_pcd[0].colors)), axis=0)
    combined_pcd = o3d.geometry.PointCloud()
    combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
    combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)

    full_path = destination_path + str(imageNumber)+ "_segmentation.ply" #on grass and original tree
    o3d.io.write_point_cloud(full_path, combined_pcd)
    #print(f"PC_visual saved to {full_path}")
    # Check if saving was OK
    point_cloud = o3d.io.read_point_cloud(full_path)
    #o3d.visualization.draw_geometries([point_cloud], window_name = "SAVED Point Cloud")

    for i, plane_mesh in enumerate(plane_meshes):  
        # Generate a filename for each plane mesh (e.g., plane_1.obj, plane_2.obj, etc.)
        file_name = f"{destination_path}/plane_{i}_mesh.obj"  #it is -1 due to indexing of listOfGrass_cloud_indexes starting from 0
        
        # Save the mesh as an .obj file
        o3d.io.write_triangle_mesh(file_name, plane_mesh)
        #print(f"Saved plane mesh {i+1} as {file_name}")





def save_RemoveGrassCSV(B_A, tree, imageNumber, flag_removed_grass, len_grass_cloud_indexes_list, maxGrassPoints,index, grass_cloud_indexes_list, plane_name ):
    """
    To CSV file save next data:
        -- tree -> which tree we are working on
        -- imageNumber -> order of images in that tree
        -- flag_removed_grass -> True if there was any data removed classified as grass
        -- len_grass_cloud_indexes_list -> NUMBER OF ELEMENTS IN list of indexes that have been classified as parts of grass plane
        -- index -> which plane is THE plane whose grass is removed
        -- grass_cloud_indexes_list -> list of indexes that have been classified as parts of grass plane
        -- plane_name -> name of the plane that is used as THE plane for removing grass
    """
    #Branch_vol2
    #destination_path = "/home/user/BRANCH_v2/images/asus/" + B_A +"/E/" 
    #Branch_vol3
    destination_path = "/home/user/BRANCH_v3/asus/" + B_A +"/E/" 
    # Define the file path for CSV
    csv_file_path = os.path.join(destination_path, "AboutRemovingGrass__perTree.csv")
    #print("file_ ", destination_path)

    # Ensure the destination directory exists
    os.makedirs(destination_path, exist_ok=True)

    fields = ['tree', 'imageNumber', 'isGrassRemoved', 'lenOfGrassCloudIndexesList', 'NumberOfPointsRemovedAsGrass','IndexOfRemovedGrassFromPreviousList', 'grass_cloud_indexes_list', 'planeName']
    data = [tree, imageNumber,flag_removed_grass, len_grass_cloud_indexes_list,maxGrassPoints, index, grass_cloud_indexes_list, plane_name ]
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


def get_ColorImageFromPointCloud(InliersRepresentingTreeWithoutGrass):
    """
    Projects a 3D point cloud onto a 2D image plane using intrinsic parameters.
    """
    # # This throws out some points if  there is -v coordinate
    image_height = 640
    image_width = 480

    # Extract points and colors from the point cloud
    points = np.asarray(InliersRepresentingTreeWithoutGrass.points)
    colors = np.asarray(InliersRepresentingTreeWithoutGrass.colors)

    # Initialize an empty image
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    for i in range(points.shape[0]):
            point = points[i]
            color = (colors[i] * 255).astype(np.uint8)  # Convert color to [0, 255] range

            # Project 3D point to 2D using the intrinsic matrix
            u = int((point[0] * FX_RGB / point[2]) + CX_RGB)
            v = int((point[1] * FY_RGB / point[2]) + CY_RGB)

            # Check if the projected point is within the image bounds
            if 0 <= u < image_width and 0 <= v < image_height:
                image[v, u] = color
    return image

def get_DepthImageFromPointCloud(InliersRepresentingTreeWithoutGrass):
    """
    Projects a 3D point cloud onto a 2D image plane
    """
    # Extract points (3D coordinates) from the point cloud
    points = np.asarray(InliersRepresentingTreeWithoutGrass.points)

    # Initialize an empty depth image (2D array)
    image_height = 640
    image_width = 480
    depth_image = np.zeros((image_height, image_width), dtype=np.float32)

    # Project each 3D point onto the 2D image plane
    for point in points:
        x, y, z = point

        # Project the 3D point to 2D using the intrinsic camera matrix
        u = int((x * FX_RGB / z) + CX_RGB)
        v = int((y * FY_RGB / z) + CY_RGB)

        # Store depth value (z-coordinate) at the corresponding pixel
        if 0 <= u < image_width and 0 <= v < image_height:
            depth_image[v, u] = z  # Depth image stores the Z value


    # Normalize the depth image to the range [0, 255] for visualization
    depth_image_normalized = np.uint8(cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX))

    return depth_image_normalized

def save_timeNeededForTeaser(tree, used_images,removed_grass, time_list):
    destination_path = "/home/user/BRANCH_v2/images/asus/" + B_A +"/E/" 
    csv_file_path = os.path.join(destination_path, "MDPI_times_grassRemoval"+B_A+".csv") 
    fields = ['tree', 'used_images','isGrassRemoved', 'grass_removal_time']
    data = [tree,  used_images, removed_grass,time_list ]
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

def getKoristeneSlike(tree,After):
    treeNumber = tree.split("_")[-1]
    print(treeNumber)
    listOfUsedNumbers = []
    if not After:
        fileName = "/home/user/Indexes_Before_Pruning_BranchVol2" + ".csv"
        dataFrame = pd.read_csv(fileName)
        strOfUsedNumbers = dataFrame.iloc[int(treeNumber),2]
        print(fileName)
    else:
        fileName = "/home/user/Indexes_After_Pruning_BranchVol2" + ".csv"
        dataFrame = pd.read_csv(fileName)
        strOfUsedNumbers = dataFrame.iloc[int(treeNumber),2]
    return strOfUsedNumbers

def remove_createdDirs_Before_Evening(start, top):
    #Branch_vol2
    #base_path = "/home/user/BRANCH_v2/images/asus/B/E/"
    #Branch_vol3
    base_path = "/home/user/BRANCH_v3/asus/B/E/"

    # Iterate over the tree directories
    for i in range(start,top):  # Adjust the range as needed (e.g., 100 for `tree_1_v_0000` to `tree_1_v_0099`)
        tree_dir = os.path.join(base_path, f"tree_1_V_{i:04d}/Filtered__noGrass")
        
        # Check if the directory exists
        if os.path.exists(tree_dir):
            # Remove the Filtered__noGrass directory
            shutil.rmtree(tree_dir)
            print(f"Removed: {tree_dir}")
        else:
            print(f"Directory not found: {tree_dir}")

#REMOVE 
#remove_createdDirs_Before_Evening(0,1)


# Asus camera
FX_RGB = 570.3422241210938
FY_RGB = 570.3422241210938
CX_RGB = 319.5
CY_RGB = 239.5

FX_DEPTH = 570.3422241210938
FY_DEPTH = 570.3422241210938
CX_DEPTH = 314.5
CY_DEPTH = 235.5
camera_depth=o3d.camera.PinholeCameraIntrinsic(640,480,FX_DEPTH,FX_DEPTH,CX_DEPTH,CY_DEPTH)

# Options for running RemoveGrass 
O3d_seed = np.random.seed(50)    # not used
visualisationFlag = False        # True if you want o3d images to be shown
After = True                     # After = False -> Meaning Before Pruning

starting_number = 0              # Main loop: [0,184] for row_number = 1 and pear_sort = "V"  OR [0, 34] for row_number = 5 and pear_sort = "V
save_Data = False
testing = False
saveTime = True
MeasuringTimeForMDPIOnlyImages = True

# Predefined 
row_number = 1                   # There were from 2 rows images taken: 1 and 5
pear_sort = "V"                  # croatian: "viljamovka" - pear tree


# Parameters for RANSAC plane segmentation through Grass points
distance_threshold = 0.06       # RANSAC distance threshold for plane fitting
ransac_n = 3                    # Minimum points for RANSAC to fit a plane
num_iterations = 1000           # Number of RANSAC iterations
min_inlier_points = 30          # Minimum inliers to consider a detected plane valid
angle_threshold = np.radians(20)# ±35 degrees in radians (sometimes 32 deg)
plane_size = 3.0                # Size of the plane in meters
found_grass = False

for i in range(starting_number,184): 

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

    #Branch_vol2
    path_to_tree_folder = "/home/user/BRANCH_v2/images/asus/" + B_A +"/E/" + tree 
    #Branch_vol3
    #path_to_tree_folder = "/home/user/BRANCH_v3/asus/" + B_A +"/E/" + tree 
    full_path = path_to_tree_folder +  "/angle0"
    lst = os.listdir(full_path + "/color") 
    numberOfImagesOfOneTree = len(lst)
    print(numberOfImagesOfOneTree)
    #all_PCD_One_Tree = []
    used_images = []
    removed_grass = []
    time_list = []

    if MeasuringTimeForMDPIOnlyImages:
        string_ofUsedImages = getKoristeneSlike(tree,After)
        print(string_ofUsedImages)
        

    for imageNumber in range(0, numberOfImagesOfOneTree):
        if(imageNumber in np.array(string_ofUsedImages.split(","), dtype = int) and MeasuringTimeForMDPIOnlyImages):
            
            # Load color and depth images, and calculate pointcloud
            print("Working on image: " + tree + " , number: "+ str(imageNumber))
            color_bp = o3d.io.read_image(full_path + "/color/" +str(imageNumber) + ".png")
            depth_bp = o3d.io.read_image(full_path+"/depth/" + str(imageNumber) + ".png")
 
            rgbd_bp = o3d.geometry.RGBDImage.create_from_color_and_depth(color_bp, depth_bp, convert_rgb_to_intensity=False)
            pcd_bp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_bp, camera_depth)
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

            grass_removal_start = time.time()

            pcd_bp.transform(rotZ(-3.14/2)) # #Rz -90
            pcd_bp.transform(rotY(3.14))    # #RY 180
            
            if(visualisationFlag):
                o3d.visualization.draw_geometries([pcd_bp], window_name = "Original Point Cloud for: "+ str(tree))


            # Visualization of the origin of the coordinate system 
            # # Step 1: Create a vertical plane as a mesh or point cloud
            # # # Define the width and height of the plane
            plane_width = 3.0   # meters
            plane_height = 3.0  # meters

            # # # Generate points for the plane on the Y-Z axis
            z = np.full((100, 1),0)  # Set X to 1 meter behind the origin
            x = np.linspace(-plane_width / 2, plane_width / 2, 10)
            y = np.linspace(-plane_height / 2, plane_height / 2, 10)

            # # # Create a mesh grid of points
            X, Y = np.meshgrid(x, y)
            points = np.vstack((X.flatten(), Y.flatten(), z.flatten())).T

            # # # Create an Open3D point cloud object for the plane
            plane_pcd = o3d.geometry.PointCloud()
            plane_pcd.points = o3d.utility.Vector3dVector(points)
            plane_pcd.paint_uniform_color([1.0, 0, 0])  # Color the plane red for visibility

            # # # Add small spheres to label the axes
            def create_text_sphere(position, color=[0, 0, 0]):
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
                sphere.paint_uniform_color(color)
                sphere.translate(position)
                return sphere

            # # # Position the labels at the ends of the coordinate frame axes
            x_label = create_text_sphere([0.6, 0, 0], color=[1, 0, 0])  # X label near the X-axis RED
            y_label = create_text_sphere([0, 0.6, 0], color=[0, 1, 0])  # Y label near the Y-axis GREEN
            z_label = create_text_sphere([0, 0, 0.6], color=[0, 0, 1])  # Z label near the Z-axis BLUE

            # # Step 2: Visualize the original point cloud with the vertical plane
            if(visualisationFlag):
                #[pcd_bp, plane_pcd, origin_frame, x_label, y_label, z_label]
                o3d.visualization.draw_geometries([pcd_bp, origin_frame],  window_name = "Original Point Cloud for: "+ str(tree)+ " with vertical plane and origin frame")
            
           

            # RANSAC plane fitting for removing Grass points 
            # # Containers for the segmented planes and their mesh representations
            segmented_planes = []                       # points in pointcloud that are part of found RANSAC plane (plane meshes) (tocke koje pripadaju ravnini)
            plane_meshes = []                           # this is a found plane (croatian Ravnina)
            grass_cloud_list = []                       # list of grass inliers
            grass_cloud_indexes_list = []               # indexed of grass inliers corresponding to original indexing in original pcd_bp
            original_points_list = [tuple(point) for point in np.asarray(pcd_bp.points)] # we are removing points that match certain plane, so this is needed for mapping remaining point indexes with original indexes
            normal_vector_list = []

            # # Define the y-axis vector
            y_axis = np.array([0, 1, 0])

            #remaining_pcd = pcd_bp          # Start with the full point cloud
            counterForGrassPlanes = 0       # Counter to see how many planes were found 

            ##Filter points at the beggining to search only the half bottom part of the tree
            points = np.asarray(pcd_bp.points)                  
            #print("numb of unfiltered points:", len(points))
            mask = points[:,1] < (np.min(points[:,1])+0.6)  # search only the bottom part of the tree (to avoid time and resource consuption) -> grass is 0.6 m long
            filtered_points = points[mask]
            #print("numb of points: ", len(filtered_points))
            remaining_pcd = o3d.geometry.PointCloud()
            remaining_pcd.points = o3d.utility.Vector3dVector(filtered_points)

            if(visualisationFlag):
                o3d.visualization.draw_geometries([remaining_pcd], window_name = "Bottom part of the tree: "+ str(tree))


            # Condition to check if grass is present by ABS DISTANCE
            # # tree usually was only wide approximately 1.5 meters, not longer! So this value was used as distance_treshold in Z axis. 
            # # Z was depth in picture

            Z_tree_length_treshold = 1.5
            abs_distance = abs(np.min(points[:,2]) - np.max(points[:,2]))
            if( abs_distance > Z_tree_length_treshold):
                if(testing):
                    print("THERE IS GRASS ", abs_distance)
                found_grass = True
                while True:
                    # # Perform RANSAC plane segmentation
                    try:
                        plane_model, inliers = remaining_pcd.segment_plane(
                            distance_threshold=distance_threshold,
                            ransac_n=ransac_n,
                            num_iterations=num_iterations
                            
                        )
                    except:
                        inliers = remaining_pcd.points

                    # # Check if the inlier count meets the minimum threshold
                    if len(inliers) < min_inlier_points:
                        break

                    # # Extract the normal of the plane
                    normal_vector = np.array(plane_model[:3])
                    normal_vector /= np.linalg.norm(normal_vector)

                    #print("normal vector: ", normal_vector)
                    # # Calculate the angle between the normal and the y-axis
                    angle_to_y_axis = np.arccos(np.dot(normal_vector, y_axis)) # Check if: ARCCOS returns positive (non-negative) solution: [0, pi]
                    #print("angle: ",angle_to_y_axis)
                    # # Extract inlier points
                    inlier_cloud = remaining_pcd.select_by_index(inliers)
                    centroid = np.mean(np.asarray(inlier_cloud.points), axis=0)
                    mean_height = centroid[1]

                    # # Check if the plane meets angle criteria
                    # # There is no need for abs(angle_to_y_axis) due to arccos always returns positive values
                    if angle_to_y_axis <= angle_threshold: # and mean_height < height_threshold: 
                        # # Color the inlier points randomly
                        if (testing):
                            print("----angle (deg): ", angle_to_y_axis*180 / np.pi)
                        color = np.random.rand(3)
                        inlier_cloud.paint_uniform_color(color)
                        segmented_planes.append(inlier_cloud)       # plane representing grass 

                        # # Create a 2x2 meter plane mesh centered at the centroid
                        plane_mesh = o3d.geometry.TriangleMesh()
                        
                        # # Create the four corners of the plane
                        half_size = plane_size / 2.0
                        points = np.array([
                            [-half_size, 0, -half_size],
                            [-half_size, 0, half_size],
                            [half_size, 0, half_size],
                            [half_size, 0, -half_size]
                        ])
                        
                        # # Rotate the plane mesh to align with the normal vector
                        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz(np.cross([0, 1, 0], normal_vector))
                        points = points @ rotation_matrix.T  # Apply rotation
                        points += centroid  # Translate to the plane's centroid

                        # # Define triangles for the square plane
                        triangles = [[0, 1, 2], [2, 3, 0]]
                        plane_mesh.vertices = o3d.utility.Vector3dVector(points)
                        plane_mesh.triangles = o3d.utility.Vector3iVector(triangles)
                        plane_mesh.paint_uniform_color(color)

            
                        # # Find corresponding indices 
                        # # ---- this is needed to find which indexes from original point clouds correspond to indexes from reamining point cloud
                        original_points_set = set(original_points_list)  # Create a set for fast lookup
                        inlier_points = np.asarray(inlier_cloud.points)

                        corresponding_original_indexes = [
                            original_points_list.index(tuple(point))
                            for point in inlier_points if tuple(point) in original_points_set
                        ]

                        # # Append the plane mesh to the visualization list
                        plane_meshes.append(plane_mesh)
                        grass_cloud_list.append(inlier_cloud)
                        grass_cloud_indexes_list.append(corresponding_original_indexes)
                        normal_vector_list.append([normal_vector[0], normal_vector[1], normal_vector[2], -np.dot(normal_vector,centroid)]) #plane equation
                        counterForGrassPlanes += 1

                    #print("-----There is number of planes found for grass: ", counterForGrassPlanes)

                    # Update the remaining point cloud by removing inliers
                    remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

                # Visualize the original point cloud and all detected plane meshes
                if(visualisationFlag):
                    o3d.visualization.draw_geometries(segmented_planes, "Segmented Planes ", width=800, height=600)
                    o3d.visualization.draw_geometries( plane_meshes, " Plane Meshes ", width=800, height=600)
                    
                    o3d.visualization.draw_geometries(segmented_planes + plane_meshes, "Segmented Planes with Plane Meshes for Grass", width=800, height=600)
                    o3d.visualization.draw_geometries([pcd_bp]+ segmented_planes +plane_meshes, "Check segmented planes  and plane mashes", width=800, height=600)
                
                flag_removed_grass = False
                
                # Check if grass_cloud_indexes_list is not empty:
                if grass_cloud_indexes_list:
                    #print("found")
                    flag_removed_grass = True
                    maxGrassPoints = 0
                    plane = o3d.geometry.TriangleMesh()
                    index_plane = 0
                    # find plane with most grass points:
                    for i in range(0, len(grass_cloud_indexes_list)):
                        if(maxGrassPoints <= len(grass_cloud_indexes_list[i])):
                            maxGrassPoints = len(grass_cloud_indexes_list[i])
                            plane =  plane_meshes[i]
                            index_plane = i
                            grass = pcd_bp.select_by_index(grass_cloud_indexes_list[i], invert = False)
                            grass.paint_uniform_color([1,0,0])  # paint grass points into RED for easier visualization
                            if(visualisationFlag):
                                o3d.visualization.draw_geometries([plane, grass_cloud_list[i]], "Visualize planes once again", width=800, height=600)
                                o3d.visualization.draw_geometries([plane, grass], "Original indexes checker", width=800, height=600)
                                o3d.visualization.draw_geometries([pcd_bp, plane, grass_cloud_list[i]], "sta je taj grass_cloud_list", width=800, height=600)
                
                    InliersRepresentingTreeWithoutGrass = pcd_bp.select_by_index(grass_cloud_indexes_list[index_plane], invert = True)
                    


                    # #remove all points in pointcloud that are considered part of the grass
                    # flat_list = [index for sublist in grass_cloud_indexes_list for index in sublist]
                    # grass_ALL = pcd_bp.select_by_index(flat_list, invert = False)
                    # InliersRepresentingTreeWithoutGrass_ALL = pcd_bp.select_by_index(flat_list, invert = True)
                    


                    # # # # # # # #

                    
                    #filter cloud to work only on the bottom of the tree to reduce time and resource consumption
                    points = np.asarray(InliersRepresentingTreeWithoutGrass.points)      # use only points what are not declared as grass
                    #print("numb of unfiltered points:", len(points))
                    mask = points[:,1] < (np.min(points[:,1])+0.7)                       # search only the bottom part of the tree (to avoid time and resource consuption) -> grass is 0.6 m long
                    filtered_points = points[mask]
                    #print("numb of points: ", len(filtered_points))
                    filtered_cloud = o3d.geometry.PointCloud()
                    filtered_cloud.points = o3d.utility.Vector3dVector(filtered_points)
                    filtered_cloud.paint_uniform_color([0,0,1])

                    # prikazati ih 
                    if( visualisationFlag):
                        o3d.visualization.draw_geometries([filtered_cloud], "Filtered cloud for bottom part of the tree where y < y_min +" + str(0.7))
                    # (NOT USED) Convert original cloud to a KD-tree for fast search
                    kdtree = o3d.geometry.KDTreeFlann(filtered_cloud)
                    
                    grass_points_ =  np.asarray(grass.points)
                    [a,b,c] = normal_vector_list[index_plane][:3]
                    d = normal_vector_list[index_plane][3]
                    # Calculate distance of each point from grass cloud to the plane
                    distances = np.abs(a * grass_points_[:, 0] + b * grass_points_[:, 1] + c * grass_points_[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)



                    ##### croatian: udaljnost svake tocke iz skupa filtriranih točaka stabala od ravnine i maknuti sve kojima je udaljenost veca od 0.5
                    ##### distance of each point from the set of filtered tree points from the plane and remove all those with a distance of 0.5
                    grass_treshold_distance = 0.35 #meters
                    distances_tree_from_plane = np.abs(a * filtered_points[:, 0] + b * filtered_points[:, 1] + c * filtered_points[:, 2] + d) / np.sqrt(a**2 + b**2 + c**2)

                    partOfGrass = np.where(distances_tree_from_plane < grass_treshold_distance)[0]
                    partOfGrass_cloud = filtered_cloud.select_by_index(partOfGrass, invert = False)

                    non_grass_indices = np.setdiff1d(np.arange(0, len(filtered_points)), partOfGrass, assume_unique=True)
                    not_partOfGrass_cloud = filtered_cloud.select_by_index(non_grass_indices, invert = False)
                    not_partOfGrass_cloud.paint_uniform_color([0, 0, 1])
                    if(visualisationFlag):
                        o3d.visualization.draw_geometries([InliersRepresentingTreeWithoutGrass, grass, not_partOfGrass_cloud], "Labeled as not grass in Filtered cloud")

                    # Visualize the nearby points
                    partOfGrass_cloud.paint_uniform_color([0, 1, 0])  # Color in green+blue for nearby points
                    if(visualisationFlag):
                        o3d.visualization.draw_geometries([InliersRepresentingTreeWithoutGrass, grass, partOfGrass_cloud], "partOfGrass points in GRASS green + red")
                
                    #find correspoinding indices from filtered cloud to original inliersrepresentintreewithoutgrass
                    grass_indices_in_original = np.where(mask)[0][partOfGrass]
                    
                    #remove indices that are part of grass; that is why there is invert = True
                    InliersRepresentingTreeWithoutGrass_ALL = InliersRepresentingTreeWithoutGrass.select_by_index(grass_indices_in_original, invert = True)
                    if(visualisationFlag):
                        o3d.visualization.draw_geometries([InliersRepresentingTreeWithoutGrass], "Tree without RANSAC detected Grass", width=800, height=600)
                        o3d.visualization.draw_geometries([InliersRepresentingTreeWithoutGrass_ALL], "Tree without --- ALL --- Grass", width=800, height=600)
                        o3d.visualization.draw_geometries([grass,partOfGrass_cloud], "Only  --- ALL --- Grass", width=800, height=600)
                    
                    if(save_Data):
                        #Save informations only if there is any planes found    
                        save_OriginPointCloudWithPlanesAndSegmentsOfGrass(path_to_tree_folder, imageNumber,[pcd_bp, grass], plane_meshes)
                        save_RemoveGrassCSV(B_A, tree, imageNumber, flag_removed_grass, len(grass_cloud_indexes_list),maxGrassPoints, index_plane, grass_cloud_indexes_list, "plane_" + str(index_plane) +"_mesh.obj" )
                        
                        color_imageFromPointCloud = get_ColorImageFromPointCloud(InliersRepresentingTreeWithoutGrass_ALL)
                        save_ColorImageFromPointCloud(path_to_tree_folder, imageNumber, color_imageFromPointCloud)    
                        depth_imageFromPointCloud = get_DepthImageFromPointCloud(InliersRepresentingTreeWithoutGrass_ALL)
                        save_DepthImageFromPointCloud(path_to_tree_folder, imageNumber, depth_imageFromPointCloud)
                        save_PointCloud(path_to_tree_folder, imageNumber,InliersRepresentingTreeWithoutGrass_ALL)
                        
                    
                else:
                    #if RANSAC has not found planes, the points that are most distanced in Z axis and closest by Y axis need to be removed
                    points = np.asarray(pcd_bp.points) 
                    mask = points[:,1] < (np.min(points[:,1])+0.6)  # search only the bottom part of the tree (to avoid time and resource consuption) -> grass is 0.6 m long
                    mask_tree = points[:,1] > (np.min(points[:,1])+0.6)
                    
                    #croatian:pronaći one točke koje imaju manju vrijednost od najdalje grane gornjeg dijela stabla
                    #find those points that have a smaller value than the furthest branch of the upper part of the tree
                    min_Z_fromTreeBranch = np.min(points[mask_tree,2])
                    grass_points = np.where(points[mask,2] < min_Z_fromTreeBranch)[0]
                    grass_indices_in_original = np.where(mask)[0][grass_points]
                    grass_testing = pcd_bp.select_by_index(grass_indices_in_original, invert=False)
                    grass_testing.paint_uniform_color([1, 0, 0])
                    InliersRepresentingTreeWithoutGrass_ALL = pcd_bp.select_by_index(grass_indices_in_original, invert=True)
                    if(visualisationFlag):
                        o3d.visualization.draw_geometries([InliersRepresentingTreeWithoutGrass_ALL, grass_testing], "Grass is red: No PLANES FOUND", width=800, height=600)
                        o3d.visualization.draw_geometries([InliersRepresentingTreeWithoutGrass_ALL], "Tree without grass: No PLANES FOUND", width=800, height=600)
                       
                    


                   # InliersRepresentingTreeWithoutGrass_ALL = copy.deepcopy(pcd_bp)
                    index_plane = -2  #meaning there is no plane, it was removed by knowing value of Z
                    if(save_Data):
                        save_RemoveGrassCSV(B_A, tree, imageNumber, True, 0, len(grass_indices_in_original), index_plane, grass_indices_in_original, "")
                        save_PointCloud(path_to_tree_folder, imageNumber,InliersRepresentingTreeWithoutGrass_ALL)

            else:
                if(testing):
                    print("NO grass ", abs_distance)
                found_grass = False
                InliersRepresentingTreeWithoutGrass_ALL = copy.deepcopy(pcd_bp)
                if(save_Data):
                    save_RemoveGrassCSV(B_A, tree, imageNumber, False, 0, 0, -1, [], "")
                    save_PointCloud(path_to_tree_folder, imageNumber,InliersRepresentingTreeWithoutGrass_ALL)
            grass_removal_stop = time.time()
            
            # visualisation for testing purposes
            if(testing):
                InliersRepresentingTreeWithoutGrass_ALL.paint_uniform_color([1, 0, 0])
                o3d.visualization.draw_geometries([pcd_bp, InliersRepresentingTreeWithoutGrass_ALL], "Tree without --- ALL --- Grass", width=800, height=600)#
                

            # for easier visually inspecting saved pcd-s
            if(save_Data):
                InliersRepresentingTreeWithoutGrass_ALL.paint_uniform_color([1, 0, 0])
                save_PointCloudForTesting(path_to_tree_folder, imageNumber,InliersRepresentingTreeWithoutGrass_ALL+pcd_bp)

            used_images.append(imageNumber)
            removed_grass.append(found_grass)
            time_list.append(grass_removal_stop - grass_removal_start)
    #save times needed for algorithm execution
    if(saveTime):
        save_timeNeededForTeaser(tree, used_images,removed_grass, time_list)


            


