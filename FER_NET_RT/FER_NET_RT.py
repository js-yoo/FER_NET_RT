import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
from model import *
import RPi.GPIO as GPIO
import time
from multiprocessing import Process

# GPIO - Board mode
output_dict = {'neutral':13, 'happiness' : 7, 'surprise': 29, 'sadness':40,
                    'anger': 12, 'fear': 24, 'disguest':''}

# Load model
def load_trained_model(model_path):
    
    model = FER_Net()
    model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage), strict=False)
    
    return model

# Recognition
def FER_Real_Time():

    # 1) Load model
    model = load_trained_model('./models/FER_trained_model.pt')
    
    emotion_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                    4: 'anger', 5: 'disguest', 6: 'fear'}

    val_transform = transforms.Compose([
        transforms.ToTensor()])

    # 2) Set Camera
    cam_path = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2  ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    cam = cv2.VideoCapture(cam_path)
    
    # Predict real-time Facial-Expression
    while True:
        
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 3) Set Detector
        FER_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        face = FER_cascade.detectMultiScale(frame)
        
        # Predict part
        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
            resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
            X = resize_frame/256
            X = Image.fromarray((X))
            X = val_transform(X).unsqueeze(0)
            
            # 4) Predict
            with torch.no_grad():
                model.eval()
                log_ps = model.cpu()(X)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                pred = emotion_dict[int(top_class.numpy())]
                
                # 5) LED Control
                output_pin=output_dict[pred]
                print("Output Pin : ",output_pin)
                
                GPIO.setmode(GPIO.BOARD)
                GPIO.setup(output_pin,GPIO.OUT,initial=GPIO.HIGH)
                print("Press CTRL+C when you want hte LED to stop blinking")
                
                time.sleep(0.5)
                GPIO.output(output_pin, GPIO.HIGH)
                print("LED ON")
                
                time.sleep(0.5)
                GPIO.output(output_pin, GPIO.LOW)
                print("LED OFF")
                
                GPIO.cleanup()  
                
            cv2.putText(frame, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

            print("Prediction : ",pred)

        # 6) Plot Prediction
        cv2.imshow('frame',frame)

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # End
    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    FER_Real_Time()