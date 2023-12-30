import cv2

# image
# img_file = "Car_Image.jpg"

#video
video = cv2.VideoCapture("Street_video.mp4")
# video = cv2.VideoCapture("video2.avi")


# pre-trained classifier
car_tracker_file = "car_detector.xml"
pedestrian_tracker_file = "haarcascade_fullbody.xml"

car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)


while True:
    
    # Read the current frame
    (read_successful, frame) = video.read()
    
    if read_successful:
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # detect cars & pedestrian
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x+1, y+1), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        
    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
    
    # Display the image with the spots
    cv2.imshow("Detector", frame) 
    key = cv2.waitKey(1)
    
    ## Stop if Q key is pressed
    if key==81 or key==113:
        break
    
# release the video capture    
video.release()

 
"""
# create opencv image
img = cv2.imread(img_file)

# convert to gratscale
black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# create car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

# detect cars
cars = car_tracker.detectMultiScale(black_n_white)

# draw rectangles around the cars
for (x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

# Display the image 
cv2.imshow("Car Detector", img)

cv2.waitKey()
"""

print ("Code Completed")