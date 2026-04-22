import os
import sys
from subprocess import call

import cv2
import numpy as np
import tensorflow.compat.v1 as tf

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.models import model



tf.disable_v2_behavior()

class SteeringAnglePredictor:
    def __init__(self,model_path):
        self.session=tf.InteractiveSession()
        self.saver=tf.train.Saver()
        self.saver.restore(self.session,model_path)
        self.smoothed_angle=0
        self.model=model

    def predict_angle(self,image):
        # radians=self.model.y.eval(feed_dict={self.model.x: [image],self.model.keep_prob: 1.0})[0][0]
        radians = self.session.run(self.model.y,feed_dict={self.model.x: [image], self.model.keep_prob: 1.0})[0][0]
        return radians * 180.0 / np.pi

    

    def smooth_angle(self,predicted_angle):
        if self.smoothed_angle==0:
            self.smoothed_angle=predicted_angle
        else:
            difference=predicted_angle-self.smoothed_angle
            if difference!=0:
                abs_differebce=abs(difference)
                scaled_difference=pow(abs_differebce,2./3.0)
                self.smoothed_angle+=(0.2*scaled_difference*(difference/abs_differebce))
        return self.smoothed_angle 
        
    def close(self):
        self.session.close()

class DrivingSimulator:
    def __init__(self,predictor,data_dir,steering_image_path,is_windows=False):
        self.predictor=predictor
        self.data_dir=data_dir
        self.steering_image=cv2.imread(steering_image_path)
        self.is_windows=is_windows

        if self.steering_image is None:
            raise ValueError(f"Could not load Steering wheel image from {steering_image_path}")
        
        # Enable alpha channel for rotation transparency
        self.steering_image = cv2.cvtColor(self.steering_image, cv2.COLOR_BGR2BGRA)

        #now my moto is to make it square such that i can resize it 
        height,width=self.steering_image.shape[:2] 
        size=max(height,width)
        # Create a square canvas
        square_img = np.zeros((size, size, 4), dtype=np.uint8)
        
        # Center the image on the square canvas
        y_offset = (size - height) // 2
        x_offset = (size - width) // 2
        square_img[y_offset:y_offset+height, x_offset:x_offset+width] = self.steering_image
        
        self.steering_image=square_img
    
    def start_simulation(self):
        i=0
        while True:
            full_image=cv2.imread(os.path.join(self.data_dir,f"{i}.jpg"))
            if full_image is None:
                # print(f"Image {i}.jpg not found in {self.data_dir}, ending simulation.")
                # Loop back to 0 to keep running or break? Standard is break.
                # But let's check input key
                if cv2.waitKey(10)==ord('q'):
                    break
                print(f"End of dataset or image not found: {i}.jpg")
                break

            if cv2.waitKey(10)==ord('q'):
                break

            resized_image=cv2.resize(full_image[-150:],(200,66))/255.0
            predicted_angle=self.predictor.predict_angle(resized_image)
            smoothed_angle=self.predictor.smooth_angle(predicted_angle)
            if not self.is_windows:
                # os.system('clear')
                call('clear')
            print(f"Predicted steering angle: {predicted_angle:.2f} degrees")
            self.display_frames(full_image,smoothed_angle)
            i+=1
        cv2.destroyAllWindows()
    
    def display_frames(self,full_image,smoothed_angle):
        #disply the main driving frame
        cv2.imshow("Driving Frame",full_image)
        #display the steering wheel frame
        cols,rows=self.steering_image.shape[:2]

        #Now the main task is to rotate the steering wheel(i am doing float diviision due to accurate precision)
        rotation_matrix=cv2.getRotationMatrix2D((cols/2,rows/2),-smoothed_angle,1)
        rotated_steering=cv2.warpAffine(self.steering_image,rotation_matrix,(cols,rows))

        #Convert to BGR for displaying usinv cv2
        steering_display_bgr=cv2.cvtColor(rotated_steering,cv2.COLOR_BGRA2BGR)
        
        #Overlay text with Predicted Angle
        font=cv2.FONT_HERSHEY_SIMPLEX
        label=f"Steering Angle: {smoothed_angle:.2f} degrees"
        cv2.putText(steering_display_bgr,label,(10,40),font,0.7,(0,255,0),2,cv2.LINE_AA)
        #Show the steering wheel seperately
        cv2.imshow("Steering Wheel",steering_display_bgr)

if __name__=="__main__":
    model_path = os.path.join(BASE_DIR, 'saved_model', 'regression_model', '50epoch', 'model.ckpt')
    data_dir = os.path.join(BASE_DIR, 'data', 'driving_dataset')
    steering_wheel_image_path = os.path.join(BASE_DIR, 'data', 'steering_wheel_image.jpg')
    
    # IF RUNNING ON WINDOWS
    is_windows = os.name == 'nt' # FALSE OTHERWISE
    
    predictor = SteeringAnglePredictor(model_path)
    simulator = DrivingSimulator(predictor, data_dir, steering_wheel_image_path, is_windows)
    
    try:
        simulator.start_simulation()
    finally:
        predictor.close()  



