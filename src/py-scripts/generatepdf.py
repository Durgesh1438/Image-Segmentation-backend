

import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import math

#from PIL import Image, ImageTk
import shutil

from PIL import Image as Img
import os
import io
#import warnings
from io import BytesIO
from fpdf import FPDF
import sys
import warnings
warnings.filterwarnings("ignore")

#warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=UserWarning)


take_photo_flag = False
coin_diameter = None
ppmm=0
df = None

options = ['Area','Perimeter','Length','Width','AspectRatio','Extent','Convex_Area','Convex_Peri',   'Solidity','Convexity','MaxDefect','AvgDefect','MinElcloDia','EquiDia','Sphericity','Eccentricity','MajAxisLen','MinAxisLen','Circularity', 'Compactness']


# create a dictionary to map the option strings to their corresponding variables
option_var_mapping = {
    "Area":"areas",  #1
    "Perimeter":"perimeters", #2
    "Length":"lengths",    #3
    "Width":"widths", #4
    "AspectRatio":"aspectratios", #5
    "Extent":"extents", #6
    "Convex_Area":"convex_areas",    #7 
    "Convex_Peri":"convex_perimeters", #8
    "Solidity":"soliditys", #9
    "Convexity":"convexitys",  #10
    "MaxDefect":"maxdefectdistances", # 11
    "AvgDefect":"avgdefectdistances", # 12
    "MinElcloDia":"minEnclosingDiameters", #13
    "EquiDia":"equi_diameters", #14
    "Sphericity":"sphericitys",  #15
    "Eccentricity":"eccentricitys",  #16
    "MajAxisLen":"major_axis_lengths", #17
    "MinAxisLen":"min_axis_lengths", #18
    "Circularity":"circularitys", #19
    "Compactness":"compactness",
}

# def calibrate(coin_diameter_entry,imagepath):
#   file_path = os.getcwd()
#   global img,ppmm,orig_img
#   # print(file_path + imagepath)
#   img=cv2.imread(file_path + imagepath)
#   orig_img= Img.fromarray(img)
 
#   image=img.copy()
#   image2=image.copy()
#   image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   blur = cv2.GaussianBlur(gray, (9,9), 0)
#   thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     # Detect circles in the image using HoughCircles
#   circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=20, minRadius=0, maxRadius=0)
#     # Ensure at least one circle was found
#   if circles is not None:
#         # Convert (x, y) coordinates and radius of circles to integers
#         circles = np.round(circles[0, :]).astype("int")

#         # Find circle with maximum radius
#         max_r = 0
#         for (x, y, r) in circles:
#             if r > max_r:
#                 max_r = r
#                 max_x = x
#                 max_y = y

#                 # Draw circle around the largest coin on image
#                 cimg=cv2.circle(image2, (max_x, max_y), max_r, (0, 255, 0), 2)

#                 # Calculate pixels per millimeter using diameter of coin
#                 rimg = cv2.resize(cimg, (350, 250))
#                 orig_img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
#                 orig_img = Img.fromarray(orig_img)
#                 #orig_img_tk = ImageTk.PhotoImage(orig_img)
#                 #orig_img_label.config(image=orig_img_tk)
#                 #orig_img_label.image = orig_img_tk
#                 coin_diameter = float(coin_diameter_entry)
#                 #coin_diameter=sys.arg[4]
#                 # print(coin_diameter)

#                 if coin_diameter is None or coin_diameter == 0:
#                     ppmm = 1
#                 ppmm = (2 * max_r) / coin_diameter
#                 print("Pixels per millimeter (ppmm): {:.2f}".format(ppmm))
#                 #ppmm_label.config(text="Pixels per millimeter (ppmm): {:.2f}".format(ppmm))
#   else:
#       ppmm = 1

def morph_analysis(ppmm,min_area_slider,max_area_slider,cluster_selection,parameterselection,imagepath,save_folder):
    global img,orig_img, img2, proc_img, areas,perimeters,lengths,widths,aspectratios,extents,convex_areas,convex_perimeters,soliditys,convexitys, maxdefectdistances, avgdefectdistances, minEnclosingDiameters, equi_diameters, sphericitys, eccentricitys, major_axis_lengths, min_axis_lengths, circularitys, compactness,df,data,data1,data2,data3

    file_path=os.getcwd()
    img=cv2.imread(file_path+imagepath)
    
    orig_img=Img.fromarray(img)
    img2=cv2.imread(file_path+imagepath)
    min_area_slider=float(min_area_slider)
    max_area_slider=float(max_area_slider)
    selected_options = parameterselection.split(',')
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the grains
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Apply morphological closing to fill gaps in the contours
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the contours of the grains
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    #min_area_threshold = min_area_slider.get()
    #max_area_threshold = max_area_slider.get()
    min_area_threshold = min_area_slider
    max_area_threshold = max_area_slider
    
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_threshold < area < max_area_threshold:
            filtered_contours.append(contour)

    if not filtered_contours:
        # Handle the case when no valid contours are found
        print("Error: No valid contours found. Adjust the area thresholds or check the image.")
        return
    
    areas = []  #1
    perimeters =[] #2
    lengths = []    #3
    widths = [] #4
    aspectratios=[] #5
    extents=[]  #6
    convex_areas = []    #7 
    convex_perimeters =[] #8
    soliditys = [] #9
    convexitys = [] #10
    maxdefectdistances = [] # 11
    avgdefectdistances = [] # 12
    minEnclosingDiameters = [] #13
    equi_diameters = [] #14
    sphericitys = [] #15
    eccentricitys = [] #16
    major_axis_lengths =[] #17
    min_axis_lengths =[] #18
    circularitys =[] #19
    compactness=[] #20
    cluster=[]

    for contour in filtered_contours:

        area = (cv2.contourArea(contour))  #1
        
        perimeter = (cv2.arcLength(contour, True)) #2
        
        (center_x, center_y), (width, height), rotation_angle = cv2.minAreaRect(contour)

        if width > height:
            length = width    
            width = height
        else:
            length = height    #3
            width = width      #4



        aspectratio=width/length    #5
        boxarea=width*height
        extent=float(area)/boxarea   #6



        #:**7. Convex Area, 8. Max Convex distance, 9. Avg Convex distance, 10. Convex Perimeter, 11. Convexity**

        hull = cv2.convexHull(contour, returnPoints=True)  # Return points of convex hull
        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
        #Convex Area
        convex_area = cv2.contourArea(cv2.convexHull(contour)) #7

        #Convex Perimeter
        convex_perimeter = cv2.arcLength(hull, closed=True) #8

        #Solidity
        solidity=area/convex_area #9

        # Convexity
        convexity=convex_perimeter/perimeter #10


        distances = []
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Find the perpendicular distance from the defect point to the line segment formed by start and end points
                distance = np.abs(np.cross(np.array(end) - np.array(start), np.array(start) - np.array(far))) / np.linalg.norm(np.array(end) - np.array(start))
                distance=round((distance) / ppmm, 2)
                distances.append(distance)
            


            #Max covex or defect distance
        if distances:
            maxdefectdistance = np.max(distances) # 11
        else:
            maxdefectdistance = 0  # Or set to a default value

    # Calculate average defect distance only if distances array is not empty
        if distances:
            avgdefectdistance = np.mean(distances) # 12
        else:
            avgdefectdistance = 0 
       
        #minEnclosingDiameter

        center, radius_circumscribed = cv2.minEnclosingCircle(contour)
        minEnclosingDiameter= radius_circumscribed*2  #13

        #Equivalent Diameter = the diameter of the circle whose area is same as the contour area. 

        equi_diameter=np.sqrt(4*area/np.pi) #14


           # Calculate the centroid and area of the contour
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Calculate the maximum inscribed circle within the contour
        radius_inscribed = float('inf')
        for point in contour:
            distance = np.sqrt((point[0][0] - cx)**2 + (point[0][1] - cy)**2)
            if distance < radius_inscribed:
                radius_inscribed = distance
        
        sphericity=radius_inscribed/radius_circumscribed #15

        #Essentricity, Major, Minor Axis
            # Calculate covariance matrix
        points = contour.reshape(-1, 2)
        cov_mat = np.cov(points.T)

        # Compute eigenvalues of the covariance matrix
        eigenvalues, _ = np.linalg.eig(cov_mat)

        # Eccentricity is the ratio of the larger eigenvalue to the smaller eigenvalue
        eccentricity_value = np.sqrt(1 - (min(eigenvalues) / max(eigenvalues))**2) #16

        # Calculate major and minor axes
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major_axis_length, minor_axis_length), angle = cv2.fitEllipse(contour)
            angle_radians = np.deg2rad(angle)

            # Calculate endpoints of major axis
            x1 = int(x + 0.5 * major_axis_length * np.cos(angle_radians))
            y1 = int(y + 0.5 * major_axis_length * np.sin(angle_radians))
            x2 = int(x - 0.5 * major_axis_length * np.cos(angle_radians))
            y2 = int(y - 0.5 * major_axis_length * np.sin(angle_radians))

            # Calculate endpoints of minor axis
            x3 = int(x + 0.5 * minor_axis_length * np.cos(angle_radians+ np.pi / 2))
            y3 = int(y + 0.5 * minor_axis_length * np.sin(angle_radians+ np.pi / 2))
            x4 = int(x - 0.5 * minor_axis_length * np.cos(angle_radians+ np.pi / 2))
            y4 = int(y - 0.5 * minor_axis_length * np.sin(angle_radians+ np.pi / 2))

            major_axis_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) #17
            minor_axis_len = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2) #18

        else:
            major_axis_len = 0
            minor_axis_len = 0

        #circularity
        circularity_value = (4 * np.pi * area) / (perimeter ** 2) #19

        # Calculate compactness
        compactness_value = (perimeter ** 2) / (4 * np.pi * area) #20
        
        # rounding value and apending
        area1 = round((area/ppmm),2)  #1
        areas.append(area1)

        perimeter1 = round((perimeter/ppmm),2) #2
        perimeters.append(perimeter1) 

        length1=round((length/ppmm),2) #3
        lengths.append(length1)

        width1=round((width/ppmm),2) #4
        widths.append(width1)

        aspectratio1 =round((aspectratio/ppmm),2)
        aspectratios.append(aspectratio1) #5

        extent1 = round((extent/ppmm),2)
        extents.append(extent1)  #6

        convex_area1=round((convex_area/ppmm),2)
        convex_areas.append(convex_area1) #7

        convex_perimeter1=round((convex_perimeter/ppmm),2)
        convex_perimeters.append(convex_perimeter1) #8

        solidity1 = round(solidity/ppmm,2)
        soliditys.append(solidity1) #9

        convexity1= round(convexity/ppmm,2)
        convexitys.append(convexity1) #10

        maxdefectdistance1=round(maxdefectdistance/ppmm,2)
        maxdefectdistances.append(maxdefectdistance1) # 11

        avgdefectdistance1 = round(avgdefectdistance/ppmm,2)
        avgdefectdistances.append(avgdefectdistance1) # 12

        minEnclosingDiameter1=round(minEnclosingDiameter/ppmm,2)
        minEnclosingDiameters.append(minEnclosingDiameter1) #13

        equi_diameter1 = round(equi_diameter/ppmm,2)
        equi_diameters.append (equi_diameter1)  #14

        sphericity1 =round(sphericity/ppmm,2)
        sphericitys.append(sphericity1) #15

        eccentricity_value1=round(eccentricity_value/ppmm,2)
        eccentricitys.append(eccentricity_value1) #16

        major_axis_len1= round(major_axis_len/ppmm,2)
        major_axis_lengths.append(major_axis_len1) #17

        minor_axis_len1=round(minor_axis_len/ppmm,2)
        min_axis_lengths.append(minor_axis_len1) #18
        
        circularity_value1=round(circularity_value/ppmm,2)
        circularitys.append(circularity_value1) #19

        compactness_value1=round(compactness_value/ppmm,2)
        compactness.append(compactness_value1) #20

    
    selected_variables = []
    # print(selected_options)
    # Check if selected_options is empty
    if not selected_options:
        selected_variables.append(lengths)  # Add lengths if no option is selected
    else:
        for option in selected_options:
            selected_variables.append(globals()[option_var_mapping[option]])
    X = np.array(selected_variables)
    X = X.reshape(-1, 1)  # Convert to a two-dimensional array with one column
    #print(selected_variables)
    #clusters_num = cluster_selection.get()
    clusters_num = cluster_selection

    if not clusters_num:
        clusters_num = 2
        
    kmeans = KMeans(n_clusters=int(clusters_num), random_state=0).fit(X)

    # Get the cluster labels for each grain
    labels = kmeans.labels_

    #print("hii")
    # Plot the clusters on the original image
    data = {
    "Grain ID": [],
    "Cluster ID": [],
    "Area": [],
    "Perimeter": [],
    "Length": [],
    "Width": [],
    "AspectRatio": [],
    "Extent": [],
    "Convex_Area": [],
    "Convex_Peri": [],
    "Solidity": [],
    }

    data1={
    "Grain ID":[],
    "Cluster ID":[],
    "Convexity": [],
    "MaxDefect": [],
    "AvgDefect": [],
    "MinElcloDia": [],
    "EquiDia": [],
    "Sphericity": [],
    "Eccentricity": [],
    "MajAxisLen": [],
    "MinAxisLen": [],
    "Circularity": [],
    "Compactness": [],
    }

    img_copy = img.copy()

    for i, contour in enumerate(filtered_contours):
        # print(i, contour)
        color = (255, 0, 0) if labels[i] == 0 else (0, 255, 0) if labels[i] == 1 else (0, 0, 255) if labels[i] == 2 else (255,255,0)
        cv2.drawContours(img_copy, [contour], -1, color, 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = 0
            cY = 0

        cv2.putText(img_copy, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cluster.append(labels[i])
        data["Grain ID"].append(i + 1)
        data["Cluster ID"].append(labels[i])
        data["Area"].append(areas[i])
        data["Perimeter"].append(perimeters[i])
        data["Length"].append(lengths[i])
        data["Width"].append(widths[i])
        data["AspectRatio"].append(aspectratios[i])
        data["Extent"].append(extents[i])
        data["Convex_Area"].append(convex_areas[i])
        data["Convex_Peri"].append(convex_perimeters[i])
        data["Solidity"].append(soliditys[i])
        
        
        df = pd.DataFrame(data)
        df.to_excel(save_folder + file_prefix + "grains.xlsx", index=False)
        #excel_file1 = save_folder + file_prefix + "grains.xlsx"
        
        data1["Grain ID"].append(i + 1)
        data1["Cluster ID"].append(0)
        data1["Convexity"].append(convexitys[i])
        data1["MaxDefect"].append(maxdefectdistances[i])
        data1["AvgDefect"].append(avgdefectdistances[i])
        data1["MinElcloDia"].append(minEnclosingDiameters[i])
        data1["EquiDia"].append(equi_diameters[i])
        data1["Sphericity"].append(sphericitys[i])
        data1["Eccentricity"].append(eccentricitys[i])
        data1["MajAxisLen"].append(major_axis_lengths[i])
        data1["MinAxisLen"].append(min_axis_lengths[i])
        data1["Circularity"].append(circularitys[i])
        data1["Compactness"].append(compactness[i])

        df1=pd.DataFrame(data1)
        df1.to_excel(save_folder + file_prefix + "grains1.xlsx", index=False)
        
        number_of_grains = len(cluster)
        number_of_clusters = len(set(cluster))  # Assuming `cluster` is a list or array
        dataex = {
        "Number of grains": [number_of_grains],
        "Number of clusters": [number_of_clusters]
        }
        # print(data)
        df3 = pd.DataFrame(dataex)
        df3.to_excel(save_folder + file_prefix + "grains2.xlsx", index=False)
        
        proc_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    
    proc_img = np.array(proc_img)
    proc_img = Img.fromarray(proc_img)
    
    excel_file1 = save_folder + file_prefix + "grains.xlsx"
    data1=pd.read_excel(excel_file1)
    excel_file2 = save_folder + file_prefix + "grains1.xlsx"
    data2=pd.read_excel(excel_file2)
    excel_file3 = save_folder + file_prefix + "grains2.xlsx"
    data3=pd.read_excel(excel_file3)





def perform_analysis(ppmm,min_area_slider,max_area_slider,cluster_selection,parameterselection,imagepath,save_folder):
    #print('RUNNING PERFORM ANALYSIS')
    global selected_options,img, orig_img, img2, proc_img, areas,perimeters,lengths,widths,aspectratios,extents,convex_areas,convex_perimeters,soliditys,convexitys, maxdefectdistances, avgdefectdistances, minEnclosingDiameters, equi_diameters, sphericitys, eccentricitys, major_axis_lengths, min_axis_lengths, circularitys, compactness,df,data,data1,data2
    selected_options = parameterselection.split(',')
    file_path = os.getcwd()

    img=cv2.imread(file_path+imagepath)
    orig_img= Img.fromarray(img)
    img2=cv2.imread(file_path+imagepath)
    min_area_slider=float(min_area_slider)
    max_area_slider=float(max_area_slider)
   
    # Convert the image to grayscale
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the grains
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Apply morphological closing to fill gaps in the contours
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the contours of the grains
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    #min_area_threshold = min_area_slider.get()
    #max_area_threshold = max_area_slider.get()
    min_area_threshold = min_area_slider
    max_area_threshold = max_area_slider
    #min_area_threshold=sys.arg[5]
    #max_area_threshold=sys.arg[6]


    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_threshold < area < max_area_threshold:
            filtered_contours.append(contour)

    if not filtered_contours:
        # Handle the case when no valid contours are found
        print("Error: No valid contours found. Adjust the area thresholds or check the image.")
        return


    # Initialize variables to store grain properties
    areas = []  #1
    perimeters =[] #2
    lengths = []    #3
    widths = [] #4
    aspectratios=[] #5
    extents=[]  #6
    convex_areas = []    #7 
    convex_perimeters =[] #8
    soliditys = [] #9
    convexitys = [] #10
    maxdefectdistances = [] # 11
    avgdefectdistances = [] # 12
    minEnclosingDiameters = [] #13
    equi_diameters = [] #14
    sphericitys = [] #15
    eccentricitys = [] #16
    major_axis_lengths =[] #17
    min_axis_lengths =[] #18
    circularitys =[] #19
    compactness=[] #20

    cluster=[]
    #print('PPMM: {:.2f}'.format(ppmm))
    # Iterate through the filtered contours and calculate the properties of each grain
    for contour in filtered_contours:

        area = (cv2.contourArea(contour))  #1
        
        perimeter = (cv2.arcLength(contour, True)) #2
        
        (center_x, center_y), (width, height), rotation_angle = cv2.minAreaRect(contour)

        if width > height:
            length = width    
            width = height
        else:
            length = height    #3
            width = width      #4



        aspectratio=width/length    #5
        boxarea=width*height
        extent=float(area)/boxarea   #6



        #:**7. Convex Area, 8. Max Convex distance, 9. Avg Convex distance, 10. Convex Perimeter, 11. Convexity**

        hull = cv2.convexHull(contour, returnPoints=True)  # Return points of convex hull
        defects = cv2.convexityDefects(contour, cv2.convexHull(contour, returnPoints=False))
        #Convex Area
        convex_area = cv2.contourArea(cv2.convexHull(contour)) #7

        #Convex Perimeter
        convex_perimeter = cv2.arcLength(hull, closed=True) #8

        #Solidity
        solidity=area/convex_area #9

        # Convexity
        convexity=convex_perimeter/perimeter #10


        distances = []
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                # Find the perpendicular distance from the defect point to the line segment formed by start and end points
                distance = np.abs(np.cross(np.array(end) - np.array(start), np.array(start) - np.array(far))) / np.linalg.norm(np.array(end) - np.array(start))
                distance=round((distance) / ppmm, 2)
                distances.append(distance)
                # Draw line from defect point to nearest point on the convex hull
                #cv2.line(img, far, (int((start[0] + end[0]) / 2), int((start[1] + end[1]) / 2)), (255, 0, 0), 1)
                # Draw defect points
                #cv2.circle(img, far, 1, [0, 0, 255], 1)


            #Max covex or defect distance
        if distances:
            maxdefectdistance = np.max(distances) # 11
        else:
            maxdefectdistance = 0  # Or set to a default value

    # Calculate average defect distance only if distances array is not empty
        if distances:
            avgdefectdistance = np.mean(distances) # 12
        else:
            avgdefectdistance = 0 
       
        #minEnclosingDiameter

        center, radius_circumscribed = cv2.minEnclosingCircle(contour)
        minEnclosingDiameter= radius_circumscribed*2  #13

        #Equivalent Diameter = the diameter of the circle whose area is same as the contour area. 

        equi_diameter=np.sqrt(4*area/np.pi) #14


           # Calculate the centroid and area of the contour
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Calculate the maximum inscribed circle within the contour
        radius_inscribed = float('inf')
        for point in contour:
            distance = np.sqrt((point[0][0] - cx)**2 + (point[0][1] - cy)**2)
            if distance < radius_inscribed:
                radius_inscribed = distance
        
        sphericity=radius_inscribed/radius_circumscribed #15

        #Essentricity, Major, Minor Axis
            # Calculate covariance matrix
        points = contour.reshape(-1, 2)
        cov_mat = np.cov(points.T)

        # Compute eigenvalues of the covariance matrix
        eigenvalues, _ = np.linalg.eig(cov_mat)

        # Eccentricity is the ratio of the larger eigenvalue to the smaller eigenvalue
        eccentricity_value = np.sqrt(1 - (min(eigenvalues) / max(eigenvalues))**2) #16

        # Calculate major and minor axes
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (x, y), (major_axis_length, minor_axis_length), angle = cv2.fitEllipse(contour)
            angle_radians = np.deg2rad(angle)

        # Calculate endpoints of major axis
        x1 = int(x + 0.5 * major_axis_length * np.cos(angle_radians))
        y1 = int(y + 0.5 * major_axis_length * np.sin(angle_radians))
        x2 = int(x - 0.5 * major_axis_length * np.cos(angle_radians))
        y2 = int(y - 0.5 * major_axis_length * np.sin(angle_radians))

        # Calculate endpoints of minor axis
        x3 = int(x + 0.5 * minor_axis_length * np.cos(angle_radians+ np.pi / 2))
        y3 = int(y + 0.5 * minor_axis_length * np.sin(angle_radians+ np.pi / 2))
        x4 = int(x - 0.5 * minor_axis_length * np.cos(angle_radians+ np.pi / 2))
        y4 = int(y - 0.5 * minor_axis_length * np.sin(angle_radians+ np.pi / 2))

        major_axis_len = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) #17
        minor_axis_len = np.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2) #18

        #circularity
        circularity_value = (4 * np.pi * area) / (perimeter ** 2) #19

        # Calculate compactness
        compactness_value = (perimeter ** 2) / (4 * np.pi * area) #20

        # rounding value and apending
        area1 = round((area/ppmm),2)  #1
        areas.append(area1)

        perimeter1 = round((perimeter/ppmm),2) #2
        perimeters.append(perimeter1) 

        length1=round((length/ppmm),2) #3
        lengths.append(length1)

        width1=round((width/ppmm),2) #4
        widths.append(width1)

        aspectratio1 =round((aspectratio/ppmm),2)
        aspectratios.append(aspectratio1) #5

        extent1 = round((extent/ppmm),2)
        extents.append(extent1)  #6

        convex_area1=round((convex_area/ppmm),2)
        convex_areas.append(convex_area1) #7

        convex_perimeter1=round((convex_perimeter/ppmm),2)
        convex_perimeters.append(convex_perimeter1) #8

        solidity1 = round(solidity/ppmm,2)
        soliditys.append(solidity1) #9

        convexity1= round(convexity/ppmm,2)
        convexitys.append(convexity1) #10

        maxdefectdistance1=round(maxdefectdistance/ppmm,2)
        maxdefectdistances.append(maxdefectdistance1) # 11

        avgdefectdistance1 = round(avgdefectdistance/ppmm,2)
        avgdefectdistances.append(avgdefectdistance1) # 12

        minEnclosingDiameter1=round(minEnclosingDiameter/ppmm,2)
        minEnclosingDiameters.append(minEnclosingDiameter1) #13

        equi_diameter1 = round(equi_diameter/ppmm,2)
        equi_diameters.append (equi_diameter1)  #14

        sphericity1 =round(sphericity/ppmm,2)
        sphericitys.append(sphericity1) #15

        eccentricity_value1=round(eccentricity_value/ppmm,2)
        eccentricitys.append(eccentricity_value1) #16

        major_axis_len1= round(major_axis_len/ppmm,2)
        major_axis_lengths.append(major_axis_len1) #17

        minor_axis_len1=round(minor_axis_len/ppmm,2)
        min_axis_lengths.append(minor_axis_len1) #18
        
        circularity_value1=round(circularity_value/ppmm,2)
        circularitys.append(circularity_value1) #19

        compactness_value1=round(compactness_value/ppmm,2)
        compactness.append(compactness_value1) #20

    
    data = {
    "Grain ID": [],
    "Cluster ID": [],
    "Area": [],
    "Perimeter": [],
    "Length": [],
    "Width": [],
    "AspectRatio": [],
    "Extent": [],
    "Convex_Area": [],
    "Convex_Peri": [],
    "Solidity": [],
    }

    data1={
    "Grain ID":[],
    "Cluster ID":[],
    "Convexity": [],
    "MaxDefect": [],
    "AvgDefect": [],
    "MinElcloDia": [],
    "EquiDia": [],
    "Sphericity": [],
    "Eccentricity": [],
    "MajAxisLen": [],
    "MinAxisLen": [],
    "Circularity": [],
    "Compactness": [],        
    }
    
    # Plot the clusters on the original image
    img_copy = img.copy()
    for i, contour in enumerate(filtered_contours):
        cv2.drawContours(img_copy, [contour], -1, (0, 0, 255), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX = 0
            cY = 0

        cv2.putText(img_copy, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        data["Grain ID"].append(i + 1)
        data["Cluster ID"].append(0)
        data["Area"].append(areas[i])
        data["Perimeter"].append(perimeters[i])
        data["Length"].append(lengths[i])
        data["Width"].append(widths[i])
        data["AspectRatio"].append(aspectratios[i])
        data["Extent"].append(extents[i])
        data["Convex_Area"].append(convex_areas[i])
        data["Convex_Peri"].append(convex_perimeters[i])
        data["Solidity"].append(soliditys[i])
        

        df = pd.DataFrame(data)

       # Export the dataframe as an Excel sheet
        df.to_excel(save_folder + file_prefix + "grains.xlsx", index=False)

        data1["Grain ID"].append(i + 1)
        data1["Cluster ID"].append(0)
        data1["Convexity"].append(convexitys[i])
        data1["MaxDefect"].append(maxdefectdistances[i])
        data1["AvgDefect"].append(avgdefectdistances[i])
        data1["MinElcloDia"].append(minEnclosingDiameters[i])
        data1["EquiDia"].append(equi_diameters[i])
        data1["Sphericity"].append(sphericitys[i])
        data1["Eccentricity"].append(eccentricitys[i])
        data1["MajAxisLen"].append(major_axis_lengths[i])
        data1["MinAxisLen"].append(min_axis_lengths[i])
        data1["Circularity"].append(circularitys[i])
        data1["Compactness"].append(compactness[i])

        df1=pd.DataFrame(data1)
        df1.to_excel(save_folder + file_prefix + "grains1.xlsx", index=False)





    proc_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    # Convert the processed image to PIL format and display it
    proc_img = np.array(proc_img)
    proc_img = Img.fromarray(proc_img)
    #proc_img_tk = ImageTk.PhotoImage(proc_img)
    #proc_img_label.config(image=proc_img_tk)
    #proc_img_label.image = proc_img_tk

    # Create a dataframe to store the grain characteristics
    
    excel_file1 = save_folder + file_prefix + "grains.xlsx"
    data1 = pd.read_excel(excel_file1)
    excel_file2=save_folder + file_prefix + "grains1.xlsx"
    data2=pd.read_excel(excel_file2)
    

def download_results():
    
    global file_path
    #file_path = filedialog.asksaveasfilename(defaultextension=".xlsx")
    shutil.copyfile(save_folder + file_prefix + "grains.xlsx", file_path)
    #top = tk.Toplevel()
    #top.title("Download Results")
    #lbl = tk.Label(top, text=f"Results saved to {file_path}")
    #lbl.pack()

def save_processed_image(filename):
    #global img
    if img is not None:
        #filename = filedialog.asksaveasfilename(defaultextension=".jpg")
        # print()
        if filename:
            cv2.imwrite(filename,img)
            #top = tk.Toplevel()
            #top.title("Download Processed Image")
            #lbl = tk.Label(top, text=f"Image saved to {filename}")
            #lbl.pack()


def color_cluster(min_area_slider,max_area_slider,cluster_selection,imagepath):
    file_path = os.getcwd()
    global img,orig_img,proc_img,data1,data2,clusters_num
    img= cv2.imread(file_path+imagepath)
   
    orig_img=Img.fromarray(img)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to segment the grains
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)

    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    # Apply morphological closing to fill gaps in the contours
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the contours of the grains
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cluster = []

    # Perform color-based clustering on the seeds
    seeds = []
    #min_area_threshold = min_area_slider.get()
    #max_area_threshold = max_area_slider.get()
    min_area_threshold = int(min_area_slider)
    max_area_threshold = int(max_area_slider)
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_threshold < area < max_area_threshold:
            filtered_contours.append(contour)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(img, mask=mask)[:3]
            seeds.append(mean_color)
    if(len(seeds) < cluster_selection):
        raise Exception('Error: Not enough samples. Try with larger area')
    # Convert the list of seed colors to NumPy array
    if seeds:
        seeds = np.array(seeds)
        clusters_num=int(cluster_selection)
        kmeans = KMeans(n_clusters=clusters_num, random_state=0)
        #print('done')
        kmeans.fit(seeds)
        labels = kmeans.labels_
        img_copy = img.copy()
        for i, contour in enumerate(filtered_contours):
            color = (255, 0, 0) if labels[i] == 0 else (0, 255, 0) if labels[i] == 1 else (0, 0, 255) if labels[i] == 2 else (255,255,0)
            cv2.drawContours(img_copy, [contour], -1, color, 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX = 0
                cY = 0
            cv2.putText(img_copy, str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cluster.append(labels[i])
        #img2 = img.copy()
        #proc_img = img2.copy()
        proc_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
        proc_img = np.array(proc_img)
        proc_img = Img.fromarray(proc_img)
        #proc_img_tk = ImageTk.PhotoImage(proc_img)
        #proc_img_label.config(image=proc_img_tk)
        #proc_img_label.image = proc_img_tk

        data = {"Grain ID": range(1, len(cluster) + 1),
                "Cluster ID": cluster}
        df1 = pd.DataFrame(data)
        df1.to_excel(save_folder + file_prefix + "grains1.xlsx", index=False)
        excel_filec1 = save_folder + file_prefix + "grains1.xlsx"
        data1 = pd.read_excel(excel_filec1)
        number_of_grains = len(cluster)
        number_of_clusters = len(set(cluster))  # Assuming `cluster` is a list or array
        data = {
            "Number of grains": [number_of_grains],
            "Number of clusters": [number_of_clusters]
        }
        df2 = pd.DataFrame(data)
        df2.to_excel(save_folder + file_prefix + "grains.xlsx", index=False)
        excel_filec2 = save_folder + file_prefix + "grains.xlsx"
        data2 = pd.read_excel(excel_filec2)
        
    else:
        print("Error: No valid seed colors found.")


vardict=[]

def update_selected_options(*args,vardict):
    selected_options.clear()
    for i, option in enumerate(options):
        if vardict[i].get():
            selected_options.append(option_var_mapping[option])

def generate_pdf_report(testid, date, variety, name, organization, email, comments,processname):
    global data1, data2,  proc_img
    file_path=os.getcwd()
    # Save original and processed images to temporary files
    orig_img_path = save_folder + file_prefix + "downloaded_original_image.png"
    proc_img_path = save_folder + file_prefix + "downloaded_processed_image.png"
    orig_img.save(orig_img_path, format="PNG")
    proc_img.save(proc_img_path, format="PNG")
    #print('second done')
    # Create a PDF document
    pdf = FPDF(format='A4')
    pdf.add_page()

    # Set font and size
    pdf.set_font("Helvetica", size=12)

    # Add content to the PDF report
    pdf.cell(0, 10, "Report generated by GraSeed - AI", ln=True)
    pdf.cell(0, 10, "Test ID: {}".format(testid), ln=True)
    pdf.cell(0, 10, "Date: {}".format(name), ln=True)
    pdf.cell(0, 10, "Test Variety: {}".format(variety), ln=True)
    pdf.cell(0, 10, "Name: {}".format(comments), ln=True)
    pdf.cell(0, 10, "Organization: {}".format(email), ln=True)
    pdf.cell(0, 10, "Email: {}".format(date), ln=True)
    pdf.cell(0, 10, "Comments: {}".format(organization), ln=True)

    pdf.ln(5)  # Add 10 units of empty space
    # Add table for data1
    header = data1.columns.tolist()
    pdf.set_fill_color(192, 192, 192)
    pdf.set_font("Helvetica", "B", size=8)

    # Calculate column widths based on the longest content
    col_widths = [max(pdf.get_string_width(str(header[i])) + 6, max(pdf.get_string_width(str(row[i])) for row in data1.values) + 6) for i in range(len(header))]

    for i, column in enumerate(header):
        pdf.cell(col_widths[i], 10, column, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_fill_color(255)
    pdf.set_font("Helvetica", size=8)

    for row in data1.values:
        for i, item in enumerate(row):
            pdf.cell(col_widths[i], 10, str(item), border=1, fill=True, align='C')
        pdf.ln()

    pdf.ln(10)  # Add 10 units of empty space
    # Add table for data2 (similar approach)

    header = data2.columns.tolist()
    pdf.set_fill_color(192, 192, 192)
    pdf.set_font("Helvetica", "B", size=8)

    # Calculate column widths based on the longest content
    col_widths = [max(pdf.get_string_width(str(header[i])) + 6, max(pdf.get_string_width(str(row[i])) for row in data2.values) + 6) for i in range(len(header))]

    for i, column in enumerate(header):
        pdf.cell(col_widths[i], 10, column, border=1, fill=True, align='C')
    pdf.ln()

    pdf.set_fill_color(255)
    pdf.set_font("Helvetica", size=8)

    for row in data2.values:
        for i, item in enumerate(row):
            pdf.cell(col_widths[i], 10, str(item), border=1, fill=True, align='C')
        pdf.ln()

    pdf.ln(10)  # Add 10 units of empty space

    if processname=='morph':
        header = data3.columns.tolist()
        pdf.set_fill_color(192, 192, 192)
        pdf.set_font("Helvetica", "B", size=8)

        # Calculate column widths based on the longest content
        col_widths = [max(pdf.get_string_width(str(header[i])) + 6, max(pdf.get_string_width(str(row[i])) for row in data3.values) + 6) for i in range(len(header))]

        for i, column in enumerate(header):
         pdf.cell(col_widths[i], 10, column, border=1, fill=True, align='C')
        pdf.ln()

        pdf.set_fill_color(255)
        pdf.set_font("Helvetica", size=8)

        for row in data3.values:
           for i, item in enumerate(row):
             pdf.cell(col_widths[i], 10, str(item), border=1, fill=True, align='C')
        pdf.ln()

        pdf.ln(10) 

    pdf.set_fill_color(255)
    pdf.set_font("Helvetica", size=8)

    # Add original image
    img_width = min(150, pdf.w - 20)  # Set the image width as needed, but not exceeding page width
    img_height = orig_img.height * img_width // orig_img.width  # Calculate proportional height

    pdf.image(orig_img_path,x=0, y=None, w=img_width, h=img_height)

    # Add some empty space
    pdf.ln(10)

    # Add text below the image
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 10, "Original Image", 0, 1, 'L')

    pdf.ln(10)


    # Add processed image
    img_width = 150  # Set the image width as needed
    img_height = proc_img.height * img_width // proc_img.width  # Calculate proportional height
    pdf.image(proc_img_path,x=7, y=None, w=img_width, h=img_height)

    # Add some empty space
    pdf.ln(10)

    # Add  text
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 10, "Processed Image", 0, 1, 'L')

    pdf.ln(10)

    # Add disclaimer text
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 10, "Disclaimer:Results are dependent on Proper Calibration",0,1,'C')

    # Save the PDF
    #pdf_file = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
    if pdf_file:
        pdf.output(file_path+pdf_file)
        print(pdf_file)
        #display_confirmation_window(pdf_file)
    else:
        print("Error: PDF report not generated")



def make_report(testid,date,variety,name,organization,email,comments,processname):
    # Create a custom multiline input dialog
    #dialog = MultilineInputDialog(None)

    # Access user inputs from the dialog
    #testid = dialog.testid
    #date = dialog.date
    #variety = dialog.variety
    #name = dialog.name
    #organization = dialog.organization
    #email = dialog.email
    #comments = dialog.comments

    # Generate the PDF report
    generate_pdf_report(testid, date, variety, name, organization, email, comments,processname)

file_prefix = sys.argv[1]
coinimageextension=sys.argv[2]
seedimageextension=sys.argv[3]
ppmm=float(sys.argv[4])
min_area_slider=int(sys.argv[5])
max_area_slider=int(sys.argv[6])
cluster_selection=int(sys.argv[7])
parameterselection=sys.argv[8]
testid=sys.argv[9]
date=sys.argv[10]
variety=sys.argv[11]
name=sys.argv[12]
organization=sys.argv[13]
email=sys.argv[14]
comments=sys.argv[15]
imagepath=sys.argv[16]
save_folder=sys.argv[17]
pdf_file=sys.argv[18]
processname=sys.argv[19]

# file_prefix = "Mahesh_"
# coinimageextension = "coin"
# seedimageextension = "seed"
# ppmm=24.8
# min_area_slider=100
# max_area_slider=1000
# cluster_selection=3
# parameterselection='lengths,widths,areas,perimeters,min_radiis'
# testid="1"
# date="23/4/23"
# variety="asf"
# name="carey"
# organization="org"
# email="care"
# comments="fghty"
# imagepath="/storage/davidenochk/original_img.png"
# save_folder="storage/davidenochk/"
# pdf_file="storage/davidenochk/final_report.pdf"

# print(ppmm,min_area_slider,max_area_slider,cluster_selection,parameterselection)

def call(ppmm,parameterselection,imagepath,min_area_slider,max_area_slider,cluster_selection,save_folder,testid,date,variety,name,organization,email,comments,processname):
    try:
        
        if processname=='measure':
          perform_analysis(ppmm,min_area_slider,max_area_slider,cluster_selection,parameterselection,imagepath,save_folder)
        if processname=='color':
          color_cluster(min_area_slider,max_area_slider,cluster_selection,imagepath)
        if processname=='morph':
            morph_analysis(ppmm,min_area_slider,max_area_slider,cluster_selection,parameterselection,imagepath,save_folder)
        #print('done')
        make_report(testid,date,variety,name,organization,email,comments,processname)
        #print('Successfully Generated')
        sys.stdout.flush()
    except Exception as error:
        print('Error: Something went wrong')
        print(error)
        sys.stdout.flush()
    
call(ppmm,parameterselection,imagepath,min_area_slider,max_area_slider,cluster_selection,save_folder,testid,date,variety,name,organization,email,comments,processname)