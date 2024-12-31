from ultralytics import YOLO
import math
import cv2
import numpy as np
import time

model = YOLO("yolo11s-pose.pt")

# Open the video source (YouTube video or local file)
video_source = "situps.mp4"  # Replace with video URL or local path
cap = cv2.VideoCapture(video_source)

# Check if the video source is valid
if not cap.isOpened():
    print("Error: Unable to open video source")
    exit()


total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
current_frame = 0
count=0
count_up_flag=0

start_time=time.time()

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("End of video stream")
        break

    # Perform pose detection on the current frame
    results = model(frame)
   
    # Process the detection results to extract key points
    for result in results:
        keypoints = result.keypoints  # Extract key points for the detected pose(s)

        if keypoints is not None:
            for person_keypoints in keypoints:
                # Assuming left and right elbow indices are 7 and 8
                np_array = person_keypoints.xy[0].numpy()# [x, y, confidence]
                
                
                

                # # Print key points for debugging
                try:
                    
                    
                    
                    left_eyes=np_array[0]
                                        
                    left_hips=np_array[11]
                                        
                    left_knees=np_array[13]
                    
                    left_ankles=np_array[15]
                    
                    print(f"Left Eye: {left_eyes}")
                     
                    if left_eyes is not None:
                                               
                        cv2.circle(frame, (int(left_eyes[0]), int(left_eyes[1])), 5, (255, 0, 0), -1)
                        cv2.circle(frame, (int(left_hips[0]), int(left_hips[1])), 5, (255, 0, 0), -1)
                        cv2.circle(frame, (int(left_knees[0]), int(left_knees[1])), 5, (255, 0, 0), -1)
                        cv2.circle(frame, (int(left_ankles[0]), int(left_ankles[1])), 5, (255, 0, 0), -1)
                     
                       
                        
                        
                      
                        
                        eyes_at_rest=520
                        base_level=frame_width
                        
                        
                        progress_bar= int(left_eyes[0])
                        progress_prct= (int(left_knees[0]) - int(left_eyes[0])) / int(left_knees[0])
                        progress_prct= int((1- progress_prct) *100)
                        
                        
                        if progress_prct > 75:
                            count_up_flag=1
                            
                        if progress_prct <= 50 and count_up_flag:
                            count=count+1
                            count_up_flag=0
                            
                        else:
                            pass
                            
                          
                        time_elapsed= int(time.time()-start_time)
                        
                        #Progress bar
                                              
                        cv2.rectangle(frame, (500, 100), 
                                    (int(left_eyes[0]),  100 ), (0, 255, 0), 8)

                        cv2.putText(frame, f"Progress {progress_prct} %", (1200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    (0, 0, 0), 2, cv2.LINE_AA, False)
                        
                        # #Insert the count
                        cv2.putText(frame, f"Count :{count}", (1200, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    (0, 0, 0), 2, cv2.LINE_AA, False)
                        
                        # #Inser the Time taken
                        cv2.putText(frame, f"Time Elapsed :{time_elapsed} s", (1500, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    (0, 0, 0), 2, cv2.LINE_AA, False)
                        
                        
                        # #Draw the lines connecting the keypoints
                        left_eyes=(int(left_eyes[0]), int(left_eyes[1]))
                                               
                        left_hips=(int(left_hips[0]), int(left_hips[1]) )
                          
                        left_knees=(int(left_knees[0]), int(left_knees[1]) )
                        left_ankles=(int(left_ankles[0]), int(left_ankles[1]) )
                                              
                        
                        cv2.line(frame, left_eyes, left_hips, (0, 255, 255), 2)
                        cv2.line(frame, left_hips, left_knees, (0, 255, 255), 2)
                        cv2.line(frame, left_knees, left_ankles, (0, 255, 255), 2)
                        
                               
                        
                        
                        
                        
                        
                        
                   
                except:
                    print("Key points not detected")
                    
                
    # Display the frame with pose annotations
    
   
    width = int(frame.shape[1] * 50 / 100)
    height = int(frame.shape[0] * 50 / 100)
    resized_frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    
    cv2.imshow("Crunch Monitoring", resized_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
        

