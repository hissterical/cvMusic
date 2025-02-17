import cv2
import mediapipe as mediapipe

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()


while True:
    success, image = cap.read()
    if not success:
        print("Failed to capture image")
        continue

    image = cv2.flip(image, 1)  # Flip for mirror effect
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for MediaPipe

    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,  # Draw on original BGR image
                hand_landmarks,
                mphands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Tracker", image)  # Show BGR image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()