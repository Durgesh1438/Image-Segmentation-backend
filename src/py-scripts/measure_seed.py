

import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import math

from PIL import Image
import shutil

from PIL import Image as Img
import os
import io
#import warnings
from io import BytesIO
import sys
#from fpdf import FPDF
import warnings
warnings.filterwarnings("ignore")


#warnings.filterwarnings("ignore", category=FutureWarning)
#warnings.filterwarnings("ignore", category=UserWarning)


take_photo_flag = False
coin_diameter = None
#ppmm=0
df = None

options = ['Area','Perimeter','Length','Width','AspectRatio','Extent','Convex_Area','Convex_Peri',   'Solidity','Convexity','MaxDefect','AvgDefect','MinElcloDia','EquiDia','Sphericity','Eccentricity','MajAxisLen','MinAxisLen','Circularity', 'Compactness']


selected_options = []
selected_variable =[]
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




def measure_seed(original_filename,min_area_slider,max_area_slider,cluster_selection,parameterselection,excel_filename1,excel_filename2,excel_filename3,output_filepath):
    file_path = os.getcwd()
    img2= cv2.imread(file_path+original_filename)
    global ppmm, areas,perimeters,lengths,widths,aspectratios,extents,convex_areas,convex_perimeters,soliditys,convexitys, maxdefectdistances, avgdefectdistances, minEnclosingDiameters, equi_diameters, sphericitys, eccentricitys, major_axis_lengths, min_axis_lengths, circularitys, compactness,df,data,data1,data2
    selected_options = parameterselection.split(',')
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

    
    img_copy = img2.copy()
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
        data["Convexity"].append(convexitys[i])
        data["MaxDefect"].append(maxdefectdistances[i])
        data["AvgDefect"].append(avgdefectdistances[i])
        data["MinElcloDia"].append(minEnclosingDiameters[i])
        data["EquiDia"].append(equi_diameters[i])
        data["Sphericity"].append(sphericitys[i])
        data["Eccentricity"].append(eccentricitys[i])
        data["MajAxisLen"].append(major_axis_lengths[i])
        data["MinAxisLen"].append(min_axis_lengths[i])
        data["Circularity"].append(circularitys[i])
        data["Compactness"].append(compactness[i])

        df = pd.DataFrame(data)

       # Export the dataframe as an Excel sheet
        df.to_excel(file_path + excel_filename3, index=False)

    processed_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    cv2.imwrite(file_path+output_filepath,processed_img)
    #print(processed_img)

    print(output_filepath)
    print(excel_filename3)
    #print(selected_variables)
    print(ppmm)



original_filename=sys.argv[1]
min_area_slider=int(sys.argv[2])
max_area_slider=int(sys.argv[3])
cluster_selection=int(sys.argv[4])
ppmm=float(sys.argv[5])
parameterselection=sys.argv[6]
excel_filename1=sys.argv[7]
excel_filename2=sys.argv[8]
excel_filename3=sys.argv[9]
output_filepath=sys.argv[10]

# print(min_area_slider,max_area_slider,cluster_selection,ppmm,parameterselection,original_filename, processed_filename, excel_filename1, excel_filename2, excel_filename3)

def call(original_filename,min_area_slider,max_area_slider,cluster_selection,parameterselection,excel_filename1,excel_filename2,excel_filename3,output_filepath):
    try:
        measure_seed(original_filename,min_area_slider,max_area_slider,cluster_selection,parameterselection,excel_filename1,excel_filename2,excel_filename3,output_filepath)
        sys.stdout.flush()
    except Exception as error:
        print(error)
        sys.stdout.flush()
        
call(original_filename,min_area_slider,max_area_slider,cluster_selection,parameterselection,excel_filename1,excel_filename2,excel_filename3,output_filepath)
