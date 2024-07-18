import cv2
import os

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image to be displayed on top of the face
overlay_image_path = os.path.abspath('./hair-png-man.png')  # Ensure the image is in the same directory or provide the full path
overlay_image = cv2.imread(overlay_image_path, cv2.IMREAD_UNCHANGED)

# Function to overlay image on top of detected face
def overlay_image_on_face(frame, face_rect, overlay_image, offset_fraction=0.8):
    x, y, w, h = face_rect
    overlay_resized = cv2.resize(overlay_image, (w, h))

    # Calculate new y-coordinate to place the overlay on top of the face
    offset = int(h * offset_fraction)
    new_y = y - h + offset
    if new_y < 0:  # Ensure the new position does not go out of frame
        new_y = 0

    # Split the overlay image into its color and alpha channels
    overlay_color = overlay_resized[:, :, :3]
    overlay_alpha = overlay_resized[:, :, 3]

    # Ensure the overlay does not go out of the frame bounds
    if new_y + h > frame.shape[0]:
        h = frame.shape[0] - new_y
        overlay_color = overlay_color[:h, :, :]
        overlay_alpha = overlay_alpha[:h, :]

    if x + w > frame.shape[1]:
        w = frame.shape[1] - x
        overlay_color = overlay_color[:, :w, :]
        overlay_alpha = overlay_alpha[:, :w]

    # Get the region of interest from the frame
    roi = frame[new_y:new_y+h, x:x+w]

    # Create a mask and inverse mask from the alpha channel
    mask = overlay_alpha / 255.0
    inverse_mask = 1.0 - mask

    # Blend the overlay with the region of interest
    for c in range(0, 3):
        roi[:, :, c] = (mask * overlay_color[:, :, c] + inverse_mask * roi[:, :, c])

    frame[new_y:new_y+h, x:x+w] = roi

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully.")

toggle_overlay = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to grayscale (Haar Cascade works with grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around each detected face and overlay image if toggled
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if toggle_overlay:
            overlay_image_on_face(frame, (x, y, w, h), overlay_image)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Capture key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        toggle_overlay = not toggle_overlay  # Toggle the overlay image
    elif key == ord('e'):
        break  # Exit the loop with 'e'

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Camera and windows have been released successfully.")
