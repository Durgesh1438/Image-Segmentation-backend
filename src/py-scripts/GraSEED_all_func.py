#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 20:16:47 2023

@author: Dr.Carey
"""

from tkinter import simpledialog
import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import shutil
from tkinter import *
from tkinter import Tk, Label
from PIL import Image as Img
import os
import io
import warnings
from io import BytesIO
from fpdf import FPDF




warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


take_photo_flag = False
coin_diameter = None
ppmm=0
df = None
#global img, proc_img,areas,lengths,widths,ma_axs,mi_axs,mi_dias,ma_dias,av_dias,perimeters,min_radiis,max_radiis,mn_densitys,roundnesss,compactnesss,elongations,roughnesss,pr_ratios,ma_thicknesss,mi_thicknesss


def get_camera_feed():
    global cap, camera_feed, stream_button, take_photo_flag, img, brightness_slider, contrast_slider,orig_img
    selected_camera = camera_selection.get()
    try:
        cap = cv2.VideoCapture(int(selected_camera), cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise ValueError("Unable to open camera")
    except Exception as e:
        error_label.config(text=str(e))
        return
    Brightness_label = tk.Label(root, text="Brightness", bg='#3D619B', font='bold 9',fg='#E9E9EB')
    Brightness_label.place(x=170,  y=95,  relx=0.01,  rely=0.01)
    Contrast_label = tk.Label(root, text="Contrast", bg='#3D619B', font='bold 9',fg='#E9E9EB')
    Contrast_label.place(x=170,  y=135,  relx=0.01,  rely=0.01)
    brightness_slider = tk.Scale(root, from_=-50, to=50, orient=tk.HORIZONTAL, length=75, resolution=1, width=10, bg='#3D619B', fg='#E9E9EB',highlightthickness=0)
    brightness_slider.set(0)
    brightness_slider.place(x=250, y=90)
    contrast_slider = tk.Scale(root, from_=0, to=200, orient=tk.HORIZONTAL, length=75, resolution=1, width=10, bg='#3D619B', activebackground='#3D619B', fg='#E9E9EB',highlightthickness=0)
    contrast_slider.set(100)
    contrast_slider.place(x=250, y=130) 
    while True:
        _, img = cap.read()
        brightness = brightness_slider.get()
        contrast = contrast_slider.get()
        img = cv2.convertScaleAbs(img, alpha=(contrast / 50), beta=(brightness - 50)) 
        rimg = cv2.resize(img, (350, 250))
        orig_img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
        orig_img = Img.fromarray(orig_img)
        orig_img_tk = ImageTk.PhotoImage(orig_img)
        orig_img_label.config(image=orig_img_tk)
        orig_img_label.image = orig_img_tk
        root.update()
        if take_photo_flag:
            cap.release()
            img_copy=img.copy()
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            rimg = cv2.resize(img_copy, (350, 250))
            orig_img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
            orig_img = Img.fromarray(orig_img)
            orig_img_tk = ImageTk.PhotoImage(orig_img)
            orig_img_label.config(image=orig_img_tk)
            orig_img_label.image = orig_img_tk
            take_photo_flag = False
            break
    return cap


def calibrate():
    global img,ppmm
    image=img.copy()
    image2=img.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Detect circles in the image using HoughCircles
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=20, minRadius=0, maxRadius=0)
    # Ensure at least one circle was found
    if circles is not None:
        # Convert (x, y) coordinates and radius of circles to integers
        circles = np.round(circles[0, :]).astype("int")
            
        # Find circle with maximum radius
        max_r = 0
        for (x, y, r) in circles:
            if r > max_r:
                max_r = r
                max_x = x
                max_y = y
               
                # Draw circle around the largest coin on image
                cimg=cv2.circle(image2, (max_x, max_y), max_r, (0, 255, 0), 2)
           
                # Calculate pixels per millimeter using diameter of coin
                rimg = cv2.resize(cimg, (350, 250))
                orig_img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
                orig_img = Img.fromarray(orig_img)
                orig_img_tk = ImageTk.PhotoImage(orig_img)
                orig_img_label.config(image=orig_img_tk)
                orig_img_label.image = orig_img_tk
                coin_diameter = float(coin_diameter_entry.get())
                print(coin_diameter)
                
                if coin_diameter is None or coin_diameter == 0:
                    ppmm = 1
                ppmm = (2 * max_r) / coin_diameter

                ppmm_label.config(text="Pixels per millimeter (ppmm): {:.2f}".format(ppmm))



def upload_image():
    global img, orig_img
    filename = filedialog.askopenfilename()
    img = cv2.imread(filename)
    rimg = cv2.resize(img, (350, 250))
    orig_img = cv2.cvtColor(rimg, cv2.COLOR_BGR2RGB)
    orig_img = Img.fromarray(orig_img)
    orig_img_tk = ImageTk.PhotoImage(orig_img)
    orig_img_label.config(image=orig_img_tk)
    orig_img_label.image = orig_img_tk


def take_photo():
    global take_photo_flag
    take_photo_flag = True

def perform_analysis():
    global img, proc_img, ppmm,df,data,data1,data2,areas,perimeters,lengths,widths,aspectratios,extents,convex_areas,convex_perimeters,soliditys,convexitys, maxdefectdistances, avgdefectdistances, minEnclosingDiameters, equi_diameters, sphericitys, eccentricitys, major_axis_lengths, min_axis_lengths, circularitys, compactness

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment the grains
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    
    # Create a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    
    # Apply morphological closing to fill gaps in the contours
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Find the contours of the grains
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area
    min_area_threshold = min_area_slider.get()
    max_area_threshold = max_area_slider.get()
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_threshold < area < max_area_threshold:
            filtered_contours.append(contour)
    
    if not filtered_contours:
        # Handle the case when no valid contours are found
        print("No valid contours found. Adjust the area thresholds or check the image.")
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

    cluster = []
    
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
        
    selected_variables = []
    
    # Check if selected_options is empty
    if not selected_options:
        selected_variables.append(lengths)  # Add lengths if no option is selected
    else:
        for option in selected_options:
            selected_variables.append(globals().get(option))
    
    X = np.array(selected_variables)
    X = X.reshape(-1, 1)  # Convert to a two-dimensional array with one column
    
    clusters_num = cluster_selection.get()
    
    if not clusters_num:
        clusters_num = 3

    kmeans = KMeans(n_clusters=int(clusters_num), random_state=0).fit(X)

    # Get the cluster labels for each grain
    labels = kmeans.labels_

    # Plot the clusters on the original image
    img_copy = img.copy() 
    for i, contour in enumerate(filtered_contours):
        color = (0, 0, 255) if labels[i] == 0 else (0, 255, 0) if labels[i] == 1 else (255, 0, 0)
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
    
    proc_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    # Convert the processed image to PIL format and display it
    proc_img = np.array(proc_img)
    proc_rimg = cv2.resize(proc_img, (350, 250))
    proc_img = Img.fromarray(proc_rimg)
    proc_img_tk = ImageTk.PhotoImage(proc_img)
    proc_img_label.config(image=proc_img_tk)
    proc_img_label.image = proc_img_tk

    # Create a dataframe to store the grain characteristics
    data = {"Grain ID": range(1, len(filtered_contours) + 1),
            "Cluster ID": cluster,
            "Area":areas,  #1
            "Perimeter":perimeters, #2
            "Length":lengths,    #3
            "Width":widths, #4
            "AspectRatio":aspectratios, #5
            "Extent":extents, #6
            "Convex_Area":convex_areas,    #7 
            "Convex_Peri":convex_perimeters, #8
            "Solidity":soliditys, #9
            "Convexity":convexitys,  #10
            "MaxDefect":maxdefectdistances, # 11
            "AvgDefect":avgdefectdistances, # 12
            "MinElcloDia":minEnclosingDiameters, #13
            "EquiDia":equi_diameters, #14
            "Sphericity":sphericitys,  #15
            "Eccentricity":eccentricitys,  #16
            "MajAxisLen":major_axis_lengths, #17
            "MinAxisLen":min_axis_lengths, #18
            "Circularity":circularitys, #19
            "Compactness":compactness, #20
            }

    df = pd.DataFrame(data)

    # Export the dataframe as an Excel sheet
    df.to_excel("grains.xlsx", index=False)
    
    data1 = {"Grain ID": range(1, len(filtered_contours) + 1),
            "Cluster ID": cluster,
            "Area":areas,  #1
            "Perimeter":perimeters, #2
            "Length":lengths,    #3
            "Width":widths, #4
            "AspectRatio":aspectratios, #5
            "Extent":extents, #6
            "Convex_Area":convex_areas,    #7 
            "Convex_Peri":convex_perimeters, #8
            "Solidity":soliditys} #9
    df1 = pd.DataFrame(data1)

    # Export the dataframe as an Excel sheet
    df1.to_excel("grains1.xlsx", index=False)
    excel_file1 = "grains1.xlsx"
    data1 = pd.read_excel(excel_file1)
    
    data2 = {"Grain ID": range(1, len(filtered_contours) + 1),
            "Convexity":convexitys,  #10
            "MaxDefect":maxdefectdistances, # 11
            "AvgDefect":avgdefectdistances, # 12
            "MinElcloDia":minEnclosingDiameters, #13
            "EquiDia":equi_diameters, #14
            "Sphericity":sphericitys,  #15
            "Eccentricity":eccentricitys,  #16
            "MajAxisLen":major_axis_lengths, #17
            "MinAxisLen":min_axis_lengths, #18
            "Circularity":circularitys, #19
            "Compactness":compactness} #20

    df2 = pd.DataFrame(data2)
    # Export the dataframe as an Excel sheet
    df2.to_excel("grains2.xlsx", index=False)
    excel_file2 = "grains2.xlsx"
    data2 = pd.read_excel(excel_file2)
    



def download_results():
    global file_path
    file_path = filedialog.asksaveasfilename(defaultextension=".xlsx")
    shutil.copyfile('grains.xlsx', file_path)
    top = tk.Toplevel()
    top.title("Download Results")
    lbl = tk.Label(top, text=f"Results saved to {file_path}")
    lbl.pack()

def save_processed_image():
    global img
    if img is not None:
        filename = filedialog.asksaveasfilename(defaultextension=".jpg")
        if filename:
            cv2.imwrite(filename,img)
            top = tk.Toplevel()
            top.title("Download Processed Image")
            lbl = tk.Label(top, text=f"Image saved to {filename}")
            lbl.pack()

        

options = ['Area','Perimeter','Length','Width','AspectRatio','Extent','Convex_Area','Convex_Peri','Solidity','Convexity','MaxDefect',
            'AvgDefect','MinElcloDia','EquiDia','Sphericity','Eccentricity','MajAxisLen','MinAxisLen','Circularity', 'Compactness']


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
    "Compactness":"compactness", #20
}

def color_cluster():
    global img, proc_img,data1,data2

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
    min_area_threshold = min_area_slider.get()
    max_area_threshold = max_area_slider.get()
    filtered_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_threshold < area < max_area_threshold:
            filtered_contours.append(contour)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(img, mask=mask)[:3]
            seeds.append(mean_color)
    
    # Convert the list of seed colors to NumPy array
    if seeds:
        seeds = np.array(seeds)
        clusters_num = cluster_selection.get()
        if not clusters_num:
            clusters_num = 3
        else:
            clusters_num = int(clusters_num)  # Convert to integer
        kmeans = KMeans(n_clusters=clusters_num, random_state=0)
        kmeans.fit(seeds)
        labels = kmeans.labels_
        img_copy = img.copy() 
        for i, contour in enumerate(filtered_contours):
            color = (0, 0, 255) if labels[i] == 0 else (0, 255, 0) if labels[i] == 1 else (255, 0, 0)
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
        proc_rimg = cv2.resize(proc_img, (350, 250))
        proc_img = Img.fromarray(proc_rimg)
        proc_img_tk = ImageTk.PhotoImage(proc_img)
        proc_img_label.config(image=proc_img_tk)
        proc_img_label.image = proc_img_tk

        data = {"Grain ID": range(1, len(cluster) + 1),
                "Cluster ID": cluster}
        df1 = pd.DataFrame(data)
        df1.to_excel("grains1.xlsx", index=False)
        excel_filec1 = "grains1.xlsx"
        data1 = pd.read_excel(excel_filec1)
        number_of_grains = len(cluster)
        number_of_clusters = len(set(cluster))  # Assuming `cluster` is a list or array
        data = {
            "Number of grains": [number_of_grains],
            "Number of clusters": [number_of_clusters]
        }
        df2 = pd.DataFrame(data)
        df2.to_excel("grains.xlsx", index=False)
        excel_filec2 = "grains.xlsx"
        data2 = pd.read_excel(excel_filec2)
    else:
        print("No valid seed colors found.")
        
def update_selected_options(*args):
    selected_options.clear()
    for i, option in enumerate(options):
        if var[i].get():
            selected_options.append(option_var_mapping[option])

class MultilineInputDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Make Report")
        labels = ["Test ID:", "Date:", "Test Variety:", "Practitioner:", "Organization:", "Phone/Email:", "Additional Information:"]
        entries = [tk.Entry(master, width=30) for _ in range(6)]
        self.comments_entry = tk.Text(master, height=5, width=30)
        self.scrollbar = tk.Scrollbar(master, command=self.comments_entry.yview)
        self.comments_entry.config(yscrollcommand=self.scrollbar.set)

        for row, (label, entry) in enumerate(zip(labels, entries)):
            tk.Label(master, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            entry.grid(row=row, column=1, pady=5)

        row += 1
        tk.Label(master, text=labels[-1]).grid(row=row, column=0, sticky=tk.W, pady=5)
        self.comments_entry.grid(row=row, column=1, pady=5)
        self.scrollbar.grid(row=row, column=2, sticky="ns", pady=5)

        # Set the focus on the first entry widget
        entries[0].focus_set()

        # Save the entry widgets for later access
        self.entries = entries

    def buttonbox(self):
        box = tk.Frame(self)

        # Override the standard "OK" button with "Make Report"
        w = tk.Button(box, text="Make Report", width=10, command=self.ok, default=tk.ACTIVE)
        w.pack(side=tk.LEFT, padx=5, pady=5)

        # Add the standard "Cancel" button
        w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def apply(self):
        # Access user inputs from the entry widgets
        self.testid = self.entries[0].get().strip() or "NA"
        self.date = self.entries[1].get().strip() or "NA"
        self.variety = self.entries[2].get().strip() or "NA"
        self.name = self.entries[3].get().strip() or "NA"
        self.organization = self.entries[4].get().strip() or "NA"
        self.email = self.entries[5].get().strip() or "NA"
        self.comments = self.comments_entry.get("1.0", tk.END).strip()

def make_report():
    # Create a custom multiline input dialog
    dialog = MultilineInputDialog(None)

    # Access user inputs from the dialog
    testid = dialog.testid
    date = dialog.date
    variety = dialog.variety
    name = dialog.name
    organization = dialog.organization
    email = dialog.email
    comments = dialog.comments

    # Generate the PDF report
    generate_pdf_report(testid, date, variety, name, organization, email, comments)

def generate_pdf_report(testid, date, variety, name, organization, email, comments):
    global data1, data2, orig_img, proc_img
    # Save original and processed images to temporary files
    orig_img_path = "original_image.png"
    proc_img_path = "processed_image.png"
    orig_img.save(orig_img_path, format="PNG")
    proc_img.save(proc_img_path, format="PNG")
    # Create a PDF document
    pdf = FPDF(format='A4')
    pdf.add_page()

    # Set font and size
    pdf.set_font("Helvetica", size=12)

    # Add content to the PDF report
    pdf.cell(0, 10, "Report generated by GraSeed - AI", ln=True)
    pdf.cell(0, 10, "Test ID: {}".format(testid), ln=True)
    pdf.cell(0, 10, "Date: {}".format(date), ln=True)
    pdf.cell(0, 10, "Test Variety: {}".format(variety), ln=True)
    pdf.cell(0, 10, "Name: {}".format(name), ln=True)
    pdf.cell(0, 10, "Organization: {}".format(organization), ln=True)
    pdf.cell(0, 10, "Email: {}".format(email), ln=True)
    pdf.cell(0, 10, "Comments: {}".format(comments), ln=True)

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
    pdf.cell(0, 10, "Disclaimer: This report is generated by GraSeed-AI; Results are dependent on Proper Calibration",0,1,'C')

    # Save the PDF
    pdf_file = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF Files", "*.pdf")])
    if pdf_file:
        pdf.output(pdf_file)
        print(f"PDF report generated: {pdf_file}")
        display_confirmation_window(pdf_file)

def display_confirmation_window(pdf_file):
    top = Tk()
    top.title("Report Saved")
    lbl = Label(top, text=f"Report saved to {pdf_file}")
    lbl.pack()
    top.mainloop()


#------------------------------------------------------------------
# Create the GUI
root = tk.Tk()
root.title("GraSEED-AI")
#root.geometry("900x700")
root.config(bg="#E9E9EB")
root.minsize(900, 670)
root.maxsize(900, 670)
#root.iconbitmap("CJ_icon.ico")


#photo = PhotoImage(file="CJ_Logo.png")

#photo_label = Label(root,image=photo)
#photo_label.place(x=20, y=5, relx=0.01, rely=0.01)



l1 = Label(root, text="GraSEED -AI", bg='#E9E9EB', font='Times 18 bold', fg='#20283E')
l1.place(x=75,  y=3,  relx=0.01,  rely=0.01)

l2 = Label(root, text="AI Powered Tool to Analyse Seed/Grain Characteristics", font='bold 10', fg='#20283E', bg='#E9E9EB')
l2.place(x=75,  y=35,  relx=0.01,  rely=0.01)

l3 = Label(root, text="v1.2 Beta", bg='#E9E9EB', font='bold 15', fg='#20283E')
l3.place(x=750,  y=35,  relx=0.01,  rely=0.01)

#-------------------------------------------------------------------
image  =  PhotoImage("img")
original_image  =  image.subsample(3,3)

frame  =  Frame(root,  width=860,  height=580,  bg='#3D619B')
frame.place(x=10,  y=65,  relx=0.01,  rely=0.01)

#frame  =  Frame(root,  width=340,  height=530,  bg='#FFFFFF')
#frame.place(x=11,  y=67,  relx=0.01,  rely=0.01)

#frame  =  Frame(root,  width=338,  height=528,  bg='#073980')
#frame.place(x=12,  y=68,  relx=0.01,  rely=0.01)

orig_img_label = tk.Label(root, bg='#3D619B')
orig_img_label.place(x=450,  y=80,  relx=0.01,  rely=0.01)

camera_dropdown_label = Label(root, text="Select Camera", bg='#3D619B', font='bold 14', fg='#E9E9EB')
camera_dropdown_label.place(x=20,  y=80,  relx=0.01,  rely=0.01)

# Create a drop-down menu to select a camera and a button to view the camera
camera_selection = StringVar(root)
camera_dropdown = OptionMenu(root, camera_selection, *range(0, 3))
camera_dropdown.config(bg="#E9E9EB",height=1,width=1)
camera_dropdown.place(x=20,  y=120,  relx=0.01,  rely=0.01)


stream_button = Button(root, text="View camera", command=get_camera_feed, bg='#E9E9EB', font='bold 9')
stream_button.place(x=80,  y=120,  relx=0.01,  rely=0.01)

# Create a button to capture an image
capture_image = tk.Button(root, text="Capture", command=take_photo,bg='#E9E9EB',font='bold 9', width=10)
capture_image.place(x=20,  y=180,  relx=0.01,  rely=0.01)

upload_button = tk.Button(root, text="Upload Image", command=upload_image,bg='#E9E9EB',font='bold 9')
upload_button.place(x=100,  y=180,  relx=0.01,  rely=0.01)

#-------------------------------------------------------------------
#Create Entry widget for coin diameter
calibrate_label = tk.Label(root, text="Calibration",bg='#3D619B',font='bold 14',fg='#E9E9EB')
calibrate_label.place(x=20,  y=220,  relx=0.01,  rely=0.01)

# Create a label and an entry box for coin diameter
coin_diameter_entry_label = tk.Label(root, text="Enter Reference Value (in mm): ", bg='#3D619B', font='bold 10',fg='#E9E9EB')
coin_diameter_entry_label.place(x=20,  y=260,  relx=0.01,  rely=0.01)

coin_diameter_entry = tk.Entry(root,width=4)
coin_diameter_entry.place(x=220,  y=260,  relx=0.01,  rely=0.01)

# Create a button for calibration
calibrate_button = tk.Button(root, text="Calibrate", command=calibrate,bg='#E9E9EB',font='bold 9')
calibrate_button.place(x=100,  y=290,  relx=0.01,  rely=0.01)

# Create a label for ppmm
ppmm_label = tk.Label(root, text="Pixels per millimeter (ppmm): ",bg='#3D619B',font='bold 10',fg='#E9E9EB')
ppmm_label.place(x=20,  y=320,  relx=0.01,  rely=0.01)

#-------------------------------------------------------------------

label = tk.Label(root, text='Cluster Analysis',bg='#3D619B',font='bold 14',fg='#E9E9EB')
label.place(x=20,  y=360,  relx=0.01,  rely=0.01)

cluster_label = tk.Label(root, text="Cluster Size",bg='#3D619B',font='bold 10',fg='#E9E9EB')
cluster_label.place(x=20,  y=400,  relx=0.01,  rely=0.01)

# Create a drop-down menu to select a cluster
cluster_selection = tk.StringVar(root)
cluster_dropdown = tk.OptionMenu(root, cluster_selection, *range(2, 5))
cluster_dropdown.config(width=1)
cluster_dropdown.config(height=1)
cluster_dropdown.config(bg="#E9E9EB")
cluster_dropdown.place(x=100,  y=400,  relx=0.01,  rely=0.01)


color_button = tk.Button(root, text="Colour Clustering", command=color_cluster, bg='#E9E9EB', font='bold 9')
color_button.place(x=20,  y=440,  relx=0.01,  rely=0.01)

button = tk.Button(root, text='Morphological Clustering',bg='#E9E9EB',font='bold 10')
button.place(x=140,  y=440,  relx=0.01,  rely=0.01)

min_area_label = tk.Label(root, text="Minimum Contour Area: ", bg='#3D619B', font='bold 9',fg='#E9E9EB')
min_area_label.place(x=160,  y=370,  relx=0.01,  rely=0.01)

max_area_label = tk.Label(root, text="Maximum Contour Area: ", bg='#3D619B', font='bold 9',fg='#E9E9EB')
max_area_label.place(x=160,  y=410,  relx=0.01,  rely=0.01)

# Creating the min_area_slider
min_area_slider = tk.Scale(root, from_=0, to=1000, orient=tk.HORIZONTAL, length=75, resolution=1, width=10,bg='#3D619B',activebackground='#3D619B',fg='#E9E9EB',highlightthickness=0)
min_area_slider.set(700)
min_area_slider.place(x=320, y=366)


# Creating the max_area_slider
max_area_slider = tk.Scale(root, from_=1000, to=5000, orient=tk.HORIZONTAL, length=75, resolution=1,width=10,bg='#3D619B',activebackground='#3D619B',fg='#E9E9EB',highlightthickness=0)
max_area_slider.set(1000)
max_area_slider.place(x=320,  y=405)


menu = tk.Menu(frame, tearoff=0)

var = []
for option in options:
    var.append(tk.BooleanVar(value=False))

for i, option in enumerate(options):
    menu.add_checkbutton(label=option, variable=var[i], command=update_selected_options)

def show_dropdown(*args):
    menu.post(button.winfo_rootx(), button.winfo_rooty()+button.winfo_height())

button.config(command=show_dropdown)

label = tk.Label(root, text='Assessment',bg='#3D619B',font='bold 14',fg='#E9E9EB')
label.place(x=20,  y=475,  relx=0.01,  rely=0.01)
# Add a button to process the image
process_image_button = tk.Button(root, text="Measure Seed/Grain", command=perform_analysis,bg='#E9E9EB',font='bold 10')
process_image_button.place(x=20, y=503,  relx=0.01,  rely=0.01)



save_button = tk.Button(root, text="Download Processed Image", command=save_processed_image,bg='#E9E9EB',font='bold 9')
save_button.place(x=20,  y=535,  relx=0.01,  rely=0.01)

download_button = tk.Button(root, text="Download Results", command=download_results,bg='#E9E9EB',font='bold 9')
download_button.place(x=20,  y=565,  relx=0.01,  rely=0.01)

l3 = Label(root,text = "Use Prescibred Camera and Console for Presicion in Results | Read Operating Manual Carefully ",bg='#3D619B',fg='#E9E9EB',font='bold 10').place(x = 25,y = 630)


#Add a label to display the processed image
proc_img_label = tk.Label(root, bg='#3D619B')
proc_img_label.place(x=450,  y=360,  relx=0.01,  rely=0.01)

# Add an error label to display error messages
error_label = tk.Label(root, fg="red",bg='#3D619B')
error_label.place(x=20,  y=597,  relx=0.01,  rely=0.01)

make_report_button = tk.Button(root, text="Make Report", command=make_report, bg='#E9E9EB', font='bold 9')
make_report_button.place(x=160, y=565, relx=0.01, rely=0.01)



root.mainloop()
root.quit()
os._exit(0)

