import math
import cv2
import numpy as np
import base64
import os
import sys
def calibrate_image(image_path,coin_diameter_entry,calibrated_filename):
    file_path = os.getcwd()
    img=cv2.imread(file_path + image_path)
    global ppmm,orig_img,result,ppmms
    image=img.copy()
    image2=image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9,9), 0)
    thresh = cv2.threshold(blur, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Detect circles in the image using HoughCircles
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 100, param1=30, param2=20, minRadius=0, maxRadius=0)
    result={}
    ppmms=[]
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
                cv2.imwrite(file_path + calibrated_filename,orig_img)
                #orig_img = Img.fromarray(orig_img)
                #orig_img_tk = ImageTk.PhotoImage(orig_img)
                #orig_img_label.config(image=orig_img_tk)
                #orig_img_label.image = orig_img_tk
                coin_diameter = float(coin_diameter_entry)
                #coin_diameter=sys.arg[4]
                #print(coin_diameter)

                if coin_diameter is None or coin_diameter == 0:
                    ppmm = 1
                ppmm = (2 * max_r) / coin_diameter
                # print("{:.2f}".format(ppmm))
                ppmms.append(ppmm)
                #ppmm_label.config(text="Pixels per millimeter (ppmm): {:.2f}".format(ppmm))
        # print('PPMM::' + ppmm + "::")
        # print('SUCCESS')
        result.__setitem__('ppmm', ppmms)
        print(ppmm)
        print(calibrated_filename)
    else:
        print('Error: No contour circles found.')

    nan = ppmm
    if math.isnan(nan):
        print('Error: PPMM is invalid')

image_path = sys.argv[1]
coin_diameter_entry=sys.argv[2]
calibrated_filename=sys.argv[3]
def call(image_path,coin_diameter_entry,calibrated_filename):
    try:
        calibrate_image(image_path,coin_diameter_entry,calibrated_filename)
        sys.stdout.flush()
    except Exception as error:
        print('Error: Something went wrong')
        print(error)
        sys.stdout.flush()
call(image_path,coin_diameter_entry,calibrated_filename)