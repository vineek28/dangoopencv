import cv2
import numpy as np 
from matplotlib import pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (r'C:\Program Files\Tesseract-OCR\tesseract.exe')
import easyocr
import keras_ocr
import math 

def get_filtered_image(image, action):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if action == 'NO_FILTER':
        filtered = img
    elif action == 'COLOURIZED':
         filtered = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif action == 'GRAYSCALE':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif action == 'BLURRED':
        width, height = img.shape[:2]
        if width > 500:
            k = (50,50)
        elif width > 200 and width <=500:
            k = (25, 25)
        else:
            k = (10,10)

        blur = cv2.blur(img, k)
        filtered = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    elif action == 'BINARY':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, filtered = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    elif action == 'INVERT':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        filtered = cv2.bitwise_not(img)

    elif action == 'EDGE_DETECTION':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3), 0)
        edges = cv2.Canny(blur, 100, 200)
        filtered = edges

    elif action == 'LINE_DETECTION':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            filtered = img    

    elif action == 'FACE_DETECTION':
        face_cascade = cv2.CascadeClassifier(r'C:\Users\vinee\OneDrive\Desktop\opencv_django\opencvEnv\Lib\site-packages\cv2\data\\haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
           cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
           img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           filtered = img

    elif action == 'SHAPE_DETECTION':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
  
        contours, _ = cv2.findContours(
           threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i = 0
  
        for contour in contours:

            if i == 0:
              i = 1
              continue

            approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)
      
            cv2.drawContours(img, [contour], 0, (0, 0, 255), 5)
  

            M = cv2.moments(contour)
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
  
            if len(approx) == 3:
                cv2.putText(img, 'Triangle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
  
            elif len(approx) == 4:
                cv2.putText(img, 'Quadilatral', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
  
            elif len(approx) == 5:
                cv2.putText(img, 'Pentagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
  
            elif len(approx) == 6:
                cv2.putText(img, 'Hexagon', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
  
            else:
                cv2.putText(img, 'circle', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


            
            filtered = img
    
    elif action == 'IMAGE_DENOISING':
        denoised_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        filtered = denoised_img

    elif action == 'IMAGE_MORPHOLOGY':
        kernel = np.ones((5,5), np.uint8)
        img =  cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        eroded_img = cv2.erode(img, kernel, iterations=1)
        dilated_img = cv2.dilate(img, kernel, iterations=1)
        filtered =  cv2.hconcat([img, eroded_img, dilated_img])
    
    elif action == 'WHITE_BALANCING':
        img_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        avg_a = np.average(img_LAB[:, :, 1])
        avg_b = np.average(img_LAB[:, :, 2])
        img_LAB[:, :, 1] = img_LAB[:, :, 1] - ((avg_a - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
        img_LAB[:, :, 2] = img_LAB[:, :, 2] - ((avg_b - 128) * (img_LAB[:, :, 0] / 255.0) * 1.2)
        balanced_image = cv2.cvtColor(img_LAB, cv2.COLOR_LAB2LRGB)
        filtered = balanced_image

    elif action == 'TEXT_DETECTION':
        reader = easyocr.Reader(['hi', 'en'], gpu=False) 
        results = reader.readtext(img, detail=1, paragraph=False)
        for (bbox, text, prob) in results:
           (tl, tr, br, bl) = bbox
           tl = (int(tl[0]), int(tl[1]))
           tr = (int(tr[0]), int(tr[1]))
           br = (int(br[0]), int(br[1]))
           bl = (int(bl[0]), int(bl[1]))
           text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
           cv2.rectangle(img, tl, br, (0, 255, 0), 2)
           cv2.putText(img, text, (tl[0], tl[1] - 10), 
           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

           filtered = img
    elif action == 'TEXT_REMOVAL':

        # def midpoint(x1, y1, x2, y2):
        #     x_mid = int((x1 + x2)/2)
        #     y_mid = int((y1 + y2)/2)
        #     return (x_mid, y_mid)
        pipeline = keras_ocr.pipeline.Pipeline()
        # def inpaint_text(img_path, pipeline):
        

            # img = keras_ocr.tools.read(img_path)
        prediction_groups = pipeline.recognize([img])
        mask = np.zeros(img.shape[:2], dtype="uint8")
        for box in prediction_groups[0]:
            x0, y0 = box[1][0]
            x1, y1 = box[1][1] 
            x2, y2 = box[1][2]
            x3, y3 = box[1][3] 
        
            x_mid0 = int((x1 + x2)/2)
            y_mid0 = int((y1 + y2)/2)
            x_mid1 = int((x0 + x3)/2)
            y_mi1 = int((y0 + y3)/2)
        
            thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
            cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
            img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)

            filtered = img



            
            
            
                    
                 

            
       



        
                
                
        
                 

        




        

    
 


           



        
    

  

        
        
        



        


    return filtered    