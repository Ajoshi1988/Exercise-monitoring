from ultralytics import YOLO
import math
import cv2
import numpy as np
import time
# model = YOLO("yolo11n-pose.pt")
# results= model(source="side_plank.jpg", show=True, conf=0.3, save=True)

# for result in results:
#     keypoints = result.keypoints
    
#     if keypoints is not None:
#         for person_keypoints in keypoints:
#             print(int(person_keypoints.xy[0][7][0]))
         


# Load the YOLO pose model
model = YOLO("yolo11s-pose.pt")

# Open the video source (YouTube video or local file)
video_source = "pullups.mp4"  # Replace with video URL or local path
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
                    left_elbow=np_array[7]
                    right_elbow=np_array[8]
                    
                    left_shoulder=np_array[5]
                    right_shoulder=np_array[6]
                    
                    left_palm=np_array[9]
                    right_palm=np_array[10]
                    
                    
                    
                    print(f"Left Shoulder: {left_shoulder}, Right Elbow: {right_shoulder}")
                    
                    if left_elbow is not None:
                        # cv2.circle(frame, (int(left_elbow[0]), int(left_elbow[1])), 5, (0, 0, 255), -1)
                        
                        cv2.circle(frame, (int(left_palm[0]), int(left_palm[1])), 5, (0, 255, 0), -1)
                        
                        cv2.circle(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 5, (255, 0, 0), -1)
                    
                    if right_elbow is not None: 
                        # cv2.circle(frame, (int(right_elbow[0]), int(right_elbow[1])), 5, (0, 0, 255), -1)
                        
                        cv2.circle(frame, (int(right_palm[0]), int(right_palm[1])), 5, (0, 255, 0), -1)
                        
                        cv2.circle(frame, (int(right_shoulder[0]), int(right_shoulder[1]) ), 5, (255, 0, 0), -1)
                        
                        
                      
                        bar_position=45
                        shoulders_at_rest=170
                        
                        progress_bar= int(right_shoulder[1])
                        progress_prct= shoulders_at_rest-int(right_shoulder[1])
                        
                        
                        if progress_prct > 95:
                            count_up_flag=1
                            
                        if progress_prct <= 60 and count_up_flag:
                            count=count+1
                            count_up_flag=0
                            
                        else:
                            pass
                            
                          
                        time_elapsed= int(time.time()-start_time)
                        
                        #Progress bar
                        cv2.rectangle(frame, (10, frame_height-10), 
                                    (10,  progress_bar ), (0, 255, 0), 8)

                        cv2.putText(frame, f"Progress {progress_prct} %", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    (0, 0, 255), 1, cv2.LINE_AA, False)
                        
                        #Inser the count
                        cv2.putText(frame, f"Count :{count}", (frame_width-130, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    (0, 0, 255), 1, cv2.LINE_AA, False)
                        
                        #Inser the Time taken
                        cv2.putText(frame, f"Time Elapsed :{time_elapsed} s", (frame_width-230, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                    (0, 0, 255), 1, cv2.LINE_AA, False)
                        
                        
                        #Draw the lines connecting the keypoints
                        left_palm=(int(left_palm[0]), int(left_palm[1]))
                        right_palm=(int(right_palm[0]), int(right_palm[1]))
                        
                        left_shoulder=(int(left_shoulder[0]), int(left_shoulder[1]) )
                        right_shoulder=(int(right_shoulder[0]), int(right_shoulder[1]) )
                        
                        cv2.line(frame, left_palm, left_shoulder, (0, 255, 255), 2)
                        cv2.line(frame, right_palm, right_shoulder, (0, 255, 255), 2)
                        
                        
                        
                        
                   
                except:
                    print("Key points not detected")
                    
                
    # Display the frame with pose annotations
    cv2.imshow("Pose Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
        

