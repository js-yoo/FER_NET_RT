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

# Load model
def load_trained_model(model_path):
    
    model = FER_Net()
    model.load_state_dict(torch.load(model_path, map_location = lambda storage, loc : storage), strict = False)
    
    return model

# Recognition
def FER_img(img_path):

    # 1) Load model
    model = load_trained_model('./models/FER_trained_model.pt')
    
    expression_dict = {0: 'neutral', 1: 'happiness', 2: 'surprise', 3: 'sadness',
                    4: 'anger', 5: 'disguest', 6: 'fear'}

    val_transform = transforms.Compose([
        transforms.ToTensor()])

    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2) Set detector
    FER_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
    face = FER_cascade.detectMultiScale(img)
    
    # 3) Predict input image
    for (x, y, w, h) in face:
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        resize_frame = cv2.resize(gray[y:y + h, x:x + w], (48, 48))
        X = resize_frame/256
        X = Image.fromarray((resize_frame))
        X = val_transform(X).unsqueeze(0)
        
        with torch.no_grad():
            model.eval()
            log_ps = model.cpu()(X)
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            pred = expression_dict[int(top_class.numpy())]
        cv2.putText(img, pred, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
        
    print("Prediction : ",pred)
    
    # 4) Plot Prediction
    def plot():
        
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.grid(False)
        plt.axis('off')
        plt.show()
    
    # 5) LED Control - for predictions

    # GPIO - Board mode
    output_dict = {'neutral':13, 'happiness' : 7, 'surprise': 29, 'sadness':40,
                    'anger': 12, 'fear': 24, 'disguest':''}
           
    output_pin = output_dict[pred]
    print("Output Pin : ",output_pin) 
    
    def led():
        
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(output_pin,GPIO.OUT,initial=GPIO.HIGH)
        print("Press CTRL+C when you want hte LED to stop blinking")

        try:         
            while True:
                time.sleep(0.5)
                GPIO.output(output_pin, GPIO.HIGH)
                print("LED ON")
                
                time.sleep(0.5)
                GPIO.output(output_pin, GPIO.LOW)
                print("LED OFF")
        finally:
            GPIO.cleanup()
    
    # Use multiprocessing - to output concurrently(plot&led)
    p1 = Process(target=plot)
    p2 = Process(target=led)
    
    p1.start()
    p2.start()
    
    p2.join() 
    p1.join()
    
    GPIO.cleanup()
    
if __name__ == "__main__":

    # Need image path to Run
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=True,
        help="path of image")
    args = vars(ap.parse_args())
    
    if not os.path.isfile(args['path']):
        print('The image path does not exists!!')
    else:
        print(args['path'])
        FER_img(args['path'])