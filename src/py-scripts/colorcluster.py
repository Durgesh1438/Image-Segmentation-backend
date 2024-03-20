import cv2
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
import os
import sys

def color_cluster(original_filename ,min_area_slider, max_area_slider, cluster_selection,  excel_filename1, excel_filename2,output_filepath):
    file_path = os.getcwd()
    img2 = cv2.imread(file_path+original_filename)
    global proc_img,data1,data2,clusters_num
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cluster = []
    seeds = []
    min_area_threshold = min_area_slider
    max_area_threshold = max_area_slider
    filtered_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area_threshold < area < max_area_threshold:
            filtered_contours.append(contour)
            mask = np.zeros(img2.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            mean_color = cv2.mean(img2, mask=mask)[:3]
            seeds.append(mean_color)

    if seeds:
        seeds = np.array(seeds)
        clusters_num = int(cluster_selection)
        kmeans = KMeans(n_clusters=clusters_num, random_state=0)
        kmeans.fit(seeds)
        labels = kmeans.labels_
        img_copy = img2.copy()
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

        # Saving the processed image with a fixed filename
        
        cv2.imwrite(file_path+output_filepath, proc_img)

        data = {"Grain ID": range(1, len(cluster) + 1), "Cluster ID": cluster}
        df1 = pd.DataFrame(data)
        df1.to_excel(file_path + excel_filename1, index=False)
        excel_filec1 = file_path + excel_filename1
        data1 = pd.read_excel(excel_filec1)
        number_of_grains = len(cluster)
        number_of_clusters = len(set(cluster))
        data = {"Number of grains": [number_of_grains], "Number of clusters": [number_of_clusters]}
        df2 = pd.DataFrame(data)
        df2.to_excel(file_path + excel_filename2, index=False)
        excel_filec2 = file_path + excel_filename2
        data2 = pd.read_excel(excel_filec2)
        print(output_filepath)
        print(excel_filename1)
        print(excel_filename2)
    else:
        print('Error: No seed found.')

# Extracting arguments from the command line
original_filename = sys.argv[1]
min_area_slider = int(sys.argv[2])
max_area_slider = int(sys.argv[3])
cluster_selection = int(sys.argv[4])
excel_filename1 = sys.argv[5]
excel_filename2 = sys.argv[6]
output_filepath=sys.argv[7]

try:
    processed_img = color_cluster(original_filename ,min_area_slider, max_area_slider, cluster_selection,  excel_filename1, excel_filename2,output_filepath)
    sys.stdout.flush()
except Exception as error:
    print('Error: Something went wrong')
    print(error)
    sys.stdout.flush()