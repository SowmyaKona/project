import cv2
import face_recognition
import os

# Directory to store captured face images
KNOWN_FACES_DIR = 'known_faces'

if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

def capture_faces(name):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit and 'c' to capture an image.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display the frame
        cv2.imshow("Capture Face", frame)
        
        # Capture face when 'c' is pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
            image_path = os.path.join(KNOWN_FACES_DIR, f"{name}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Face captured for {name}.")

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run the function to capture faces
# Example: capture_faces("JohnDoe")
