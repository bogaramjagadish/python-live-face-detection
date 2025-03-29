import cv2

# Load pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the webcam and start face detection
def detect_faces_live():
    # Capture video from the webcam
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Read each frame from the webcam
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Count the number of faces detected
        num_faces = len(faces)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Display the live face count on the screen
        text = f"Live Count: {num_faces}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (10, 50), font, 1.5, (0, 255, 0), 3, cv2.LINE_AA)

        # Display the frame with the detected faces and live count
        cv2.imshow('Live Face Detection', frame)

        # Exit the loop when the user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Start live face detection
detect_faces_live()
