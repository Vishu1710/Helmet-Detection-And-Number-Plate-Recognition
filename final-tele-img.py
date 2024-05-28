

import argparse
from PIL import Image

import sys
import time

import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils
import serial
import time
import pytesseract
import re


image_path = "teat1.jpeg"


pytesseract.pytesseract.tesseract_cmd = 'C:\\Tesseract-OCR\\tesseract'
user_patterns_config = '--tessdata-dir    "C:\\Tesseract-OCR\\tessdata" --user-patterns "C:\\Users\\639pr\\project2024\\helmet\\src\\helmet-detection\\my.patterns"'


import csv
from datetime import datetime


import re
import uuid


import cv2
import argparse
import numpy as np


class User:

    def __init__(self, name, reg, chatid):
        self.name = name
        self.reg = reg
        self.chatid = chatid

users = []


name1 = User("VISHNU", "KL10Z3333", "1406533227")
name2 = User("YASHIKA", "CH07X1800", "5883371392")
name3 = User("KOMAL", "reg3", "1445597382")
name4 = User("HRITIK", "reg4", "985723139")
name5 = User("YASHIKA", "CH07X18005", "5883371392")


users.append(name1)
users.append(name2)
users.append(name3)
users.append(name4)
users.append(name5)


import requests

def getChatId(reg):
    for usr in users:
        if usr.reg == reg:
            return usr.chatid
        
    return None



def sendNotification(reg):

    chatid = getChatId(reg)
    if chatid is None:
        print("contact information is not available!")
        return
    
    messaage = "Dear User  \n  This is to inform you that an e-challan has been generated against you for the violation of not wearing a helmet while driving."
    api_key = "7026679152:AAEwpY113N_650eFdL7gsIBOHXCrJzJI-H4"
    url  = "https://api.telegram.org/bot"+api_key+"/sendMessage?chat_id="+chatid+"&text="+messaage
    

    resp = requests.get(url)
    print(resp.text)



 


# sendNotification("reg1")
# sendNotification("reg2")
# sendNotification("reg3")
# sendNotification("reg4")

# exit(0)



classes = None

with open("yolov3.txt", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)





def generate_unique_name():
    unique_name = str(uuid.uuid4())
    current_datetime = datetime.now().strftime('%Y%m%d%H%M%S')
    return f"{current_datetime}_{unique_name}"


def update_csv_file(registration, image_path):
    # Define CSV file path and column headers
    csv_file_path = 'challans.csv'
    fieldnames = ['registration', 'date', 'time', 'lat', 'long', 'image']

    # Get the current date and time
    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M:%S')

    # Default values for lat and long
    default_lat = 0.0
    default_long = 0.0

    # Create or open the CSV file
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # If the file is empty, write the header
        if csv_file.tell() == 0:
            writer.writeheader()

        # Write a new row with the provided data
        writer.writerow({
            'registration': registration,
            'date': current_date,
            'time': current_time,
            'lat': default_lat,
            'long': default_long,
            'image': image_path
        })

    



def getValidateNumber(input_text):

    input_text = input_text.upper().replace(" ","")
    
    # remove non alpha numaric
    input_text = re.sub('[^0-9a-zA-Z]+', '*', input_text)
    # Regular expression pattern
    pattern = "[A-Z]{2}\d{2}([0-9A-Z]{2}|[0-9A-Z]{1})\d{4}"

    

    # Search for the pattern
    match = re.search(pattern, input_text)

    if match:
        # If a match is found, print the matched parts
        print('Matched Text:', match.group(0))     
        return match.group(0)
    else:
        print('No match found.')
        return ""


def detectHelmate(image, model):
    # Initialize the object detection model
    options = ObjectDetectorOptions(
        num_threads=8,
        score_threshold=0.3,
        max_results=3,
        enable_edgetpu= bool(False))
      
    detector = ObjectDetector(model_path=model, options=options)
    image =  image#cv2.imread(image_path)
    detections = detector.detect(image)
    image, detected_classes = utils.visualize(image, detections)
    print("detectHelmate [] detected_classes:",detected_classes)
    return image, detected_classes



def getNumberPlate(image, model):
    # Initialize the object detection model
    options = ObjectDetectorOptions(
        num_threads=8,
        score_threshold=0.4,
        max_results=1,
        enable_edgetpu= bool(False))
      
    detector = ObjectDetector(model_path=model, options=options)

    detections = detector.detect(image)
    
    image, crop_img , final_catagory = utils.visualize_and_crop(image, detections)
    print("getNumberPlate [] final_catagory:",final_catagory)

    if len(detections) > 0:
        return image, crop_img
    
    return image, None








# image = cv2.imread(image_path)
# cv2.imshow('Captured Image0', image)




def scan_for_motorcycle_and_person(image):
    image = image.copy()
    #image = cv2.resize(image,(640,360),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392


    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    motorcycle = False
    person = False


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

        #print(class_ids[i])
        print(classes[class_ids[i]])
       

        if classes[class_ids[i]] == "motorcycle":
            print("motorcycle detectd")
            motorcycle = True

        if classes[class_ids[i]] == "person":
            print("person detectd") 
            person = True


    # print("motorcycle:",motorcycle)
    # print("person:",person)

    if person and motorcycle:
        print("detect for helmet")
        return True, image
    

    return False, image




frame = cv2.imread('7.jpg')
        
image = frame     

capture = False
cv2.imshow('Live Image', frame)
motorcycle, image_morocycle = scan_for_motorcycle_and_person(image=image)
detection_timestamp = time.time()
cv2.imshow('image_morocycle', image_morocycle)

if motorcycle:
    
    finalimage, detected_classes = detectHelmate(frame,'helmate-tflite.tflite0')
    print("res: ",detected_classes)

    if "Without Helmet" in detected_classes:
        print("'Without Helmet FOUND")
        finalimage, numberplate = getNumberPlate(image,'hsrp_number_plate-tflite1.tflite4')
        image_path_name = "challans_images//"+generate_unique_name()
        cv2.imwrite(image_path_name+".jpg", finalimage)
        detection_timestamp = time.time()
    else:
        numberplate = None


    
    cv2.imshow('Detection', finalimage)
    if not numberplate is None:
        numberplate = cv2.cvtColor(numberplate, cv2.COLOR_BGR2GRAY)
        #tophat = cv2.morphologyEx(numberplate, cv2.MORPH_TOPHAT, rectKernel)
        cv2.imshow('Number-plate', numberplate)
        im_pil = Image.fromarray(numberplate)

        textStr = pytesseract.image_to_string( im_pil, lang='eng', config=user_patterns_config) 
        print('textStr tessdata_dir_config:',textStr)

        textStr = textStr.replace(" ","")
        
        reg_number = getValidateNumber(textStr)
        print("Refine:",reg_number)
        if not reg_number is None:
            print("save it to csv file")
            update_csv_file(registration=reg_number, image_path=image_path_name)
            sendNotification(reg_number)
        
            
while True:
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q") or key == ord("Q"):
        break

                
cv2.destroyAllWindows()
                
    

    
    
    
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    