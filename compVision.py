import cv2
import mediapipe as mp
import math

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
    

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks)==2:
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]

        thumb1 = hand1.landmark[mphands.HandLandmark.THUMB_TIP]
        index1 = hand1.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
        thumb2 = hand2.landmark[mphands.HandLandmark.THUMB_TIP]
        index2 = hand2.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]


        h,w, _ = image.shape
        thumb1x, thumb1y = int(thumb1.x*w), int(thumb1.y*h)
        index1x, index1y = int(index1.x*w), int(index1.y*h)
        thumb2x, thumb2y = int(thumb2.x*w), int(thumb2.y*h)
        index2x, index2y = int(index2.x*w), int(index2.y*h)
        mid1x = (thumb1x + index1x) // 2
        mid1y = (thumb1y + index1y) // 2
        mid2x = (thumb2x + index2x) // 2
        mid2y = (thumb2y + index2y) // 2


        cv2.circle(image, (thumb1x, thumb1y), 10, (0, 255, 0), -1)  # Green for thumb1
        cv2.circle(image, (index1x, index1y), 10, (0, 0, 255), -1)  # Red for index1
        cv2.circle(image, (thumb2x, thumb2y), 10, (0, 255, 0), -1)  # Green for thumb2
        cv2.circle(image, (index2x, index2y), 10, (0, 0, 255), -1)  # Red for index2

        cv2.line(image, (thumb1x, thumb1y), (index1x, index1y), (255, 255, 255), 2)
        cv2.line(image, (thumb2x, thumb2y), (index2x, index2y), (255, 255, 255), 2)
        cv2.line(image, (mid1x, mid1y), (mid2x, mid2y), (255, 255, 0), 2) 

        dist1 = math.sqrt((thumb1x - index1x) ** 2 + (thumb1y - index1y) ** 2)
        dist2 = math.sqrt((thumb2x - index2x) ** 2 + (thumb2y - index2y) ** 2)
        mid_dist = math.sqrt((mid2x - mid1x) ** 2 + (mid2y - mid1y) ** 2)

        cv2.putText(image, f"{int(dist1)}", ((thumb1x + index1x) // 2, (thumb1y + index1y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"{int(dist2)}", ((thumb2x + index2x) // 2, (thumb2y + index2y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"{int(mid_dist)}", ((mid1x + mid2x) // 2, (mid1y + mid2y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Hand Tracker", image)  # Show BGR image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()