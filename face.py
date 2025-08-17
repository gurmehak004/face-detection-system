import cv2  # OpenCV library for computer vision tasks

# Step 1: Load the pre-trained face detector model (Haar Cascade)
# OpenCV comes with many pre-trained models for face, eyes, etc., detection.
# We'll use the 'haarcascade_frontalface_default.xml' for detecting faces.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Step 2: Initialize the webcam for video capture
# VideoCapture(0) opens the default webcam. If you have multiple webcams, change the index (0, 1, 2, etc.)
cap = cv2.VideoCapture(0)

# Check if the webcam opened correctly
if not cap.isOpened():
    print("Error: Ciould not open webcam.")
    exit()

# Step 3: Process frames from the webcam in real-time
while True:
    # Capture each frame
    ret, frame = cap.read()

    # If frame not captured correctly, continue to next iteration
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale
    # Face detection generally works better on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Step 4: Detect faces in the grayscale frame
    # detectMultiScale method detects objects (faces in this case).
    # scaleFactor: how much the image size is reduced at each image scale (1.1 means 10% reduction)
    # minNeighbors: how many neighbors each candidate rectangle should have to retain it
    # minSize: Minimum possible object size. Objects smaller than this are ignored.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Step 5: Draw rectangles around detected faces
    # Loop through all the detected faces and draw a rectangle around each one
    for (x, y, w, h) in faces:
        # (x, y) is the top-left corner, (w, h) is the width and height of the rectangle
        # We use color (255, 0, 0) which is blue in BGR format, and a thickness of 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Step 6: Additional visual information (optional)
        # You can add a label on top of the rectangle showing that it's a face
        cv2.putText(frame, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Step 7: Display the frame with the detected faces
    cv2.imshow('Real-Time Face Detection', frame)

    # Step 8: Add the ability to quit the program
    # cv2.waitKey(1) waits for 1ms for a key press; if 'q' is pressed, the loop will break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 9: Release the webcam and close all OpenCV windows
# When everything is done, release the capture object
cap.release()

# Close all the windows opened by OpenCV
cv2.destroyAllWindows()
