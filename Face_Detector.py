import cv2

# Load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in 
# img = cv2.imread('kristen_stewart.jpg')

# To capture video from webcam
# instead of 0 pass the local video here
webcam = cv2.VideoCapture(0)  

## Interate forever over frames
while True:
    
    ## Read the current frame
    successful_frame_read, frame = webcam.read()
    
    # must conver to grayscale
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect Face
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
    
    
    # Draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)
    
    ## Stop if Q key is pressed
    if key==81 or key==113:
        break
    

## Release the VideoCapture object
webcam.release()
  
print("code Completed")



    
"""
key = cv2.waitKey(1)


# must conver to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect Face
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)

# Draw rectangles around the faces
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# print(face_coordinates)

# Display the image with the faces
cv2.imshow('Face Detector', img)
cv2.waitKey()


print("code Completed")
"""