import cv2
import mediapipe as mp

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Function to check if a finger is open
def is_finger_open(landmarks, finger_landmark_indices):
    base = landmarks[finger_landmark_indices[0]]
    tip = landmarks[finger_landmark_indices[1]]
    return tip.y < base.y

# Open the camera
camera = cv2.VideoCapture(0)

# Create a resizable window
cv2.namedWindow('Hand Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Hand Detection', 800, 600)  # Set the initial window size

# Predefined projection model: Thumbs up (only the thumb is open)
expected_projection = {
    'thumb': 'open',
    'index': 'closed',
    'middle': 'closed',
    'ring': 'closed',
    'pinky': 'closed'
}

# Load or create an image showing the example "thumbs up" projection
# For this example, we'll just assume you have a pre-existing image of a "thumbs up"
# Example: "thumbs_up_projection.png"
example_projection_image = cv2.imread("thumbs_up.jpg")  # Make sure the image exists in the folder

# Resize it to fit into the screen as a reference
example_projection_image = cv2.resize(example_projection_image, (75, 150))

while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (MediaPipe works with RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Initialize variables to track finger status
    finger_status = {
        'thumb': 'closed',
        'index': 'closed',
        'middle': 'closed',
        'ring': 'closed',
        'pinky': 'closed'
    }

    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check finger status (open/closed)
            landmarks = hand_landmarks.landmark

            # Check thumb
            if is_finger_open(landmarks, [1, 4]):
                finger_status['thumb'] = 'open'

            # Check index finger
            if is_finger_open(landmarks, [5, 8]):
                finger_status['index'] = 'open'

            # Check middle finger
            if is_finger_open(landmarks, [9, 12]):
                finger_status['middle'] = 'open'

            # Check ring finger
            if is_finger_open(landmarks, [13, 16]):
                finger_status['ring'] = 'open'

            # Check pinky
            if is_finger_open(landmarks, [17, 20]):
                finger_status['pinky'] = 'open'

            # Compare the current finger status with the expected projection
            if finger_status == expected_projection:
                text_annotation = "Correct Projection: Thumbs Up!"
            else:
                text_annotation = "Wrong Projection"

            # Add the annotation to the frame
            cv2.putText(frame, text_annotation, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Overlay the example projection image below the text
    frame[40:190, 10:85] = example_projection_image

    # Display the output frame
    cv2.imshow('Hand Detection', frame)

    k = cv2.waitKey(10)
    if k == 27:  # Press ESC to exit
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
