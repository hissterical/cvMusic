from pyo import *
import cv2
import mediapipe as mp
import math

# =============================
# Audio Setup with pyo
# =============================

# Initialize the audio server.
s = Server().boot()
s.start()

# Load the audio file (looping for continuous playback)
sf = SfPlayer("heads.wav", speed=1, loop=True, mul=0.5)

# Pitch shifting alternative using Harmonizer and frequency shifting
harm = Harmonizer(sf, transpo=0).out()
freq_shift = FreqShift(sf, shift=0).out()

# Define functions for audio control.
def set_volume(volume: float):
    sf.mul = max(0.0, min(volume, 1.0))  # Clamp between 0.0 and 1.0

def set_speed(speed: float):
    sf.speed = max(0.1, speed)  # Prevent zero or negative speed

def set_pitch(semitones: float):
    harm.transpo = semitones  # Pitch shifting via semitones

def set_frequency_shift(shift_hz: float):
    freq_shift.shift = shift_hz

# =============================
# Smoothing Class
# =============================
class SmoothValue:
    """
    Smooths rapid changes by blending the new value with the previous value.
    alpha: 0.0 (no update) to 1.0 (no smoothing).
    """
    def __init__(self, alpha=0.1):
        self.value = None
        self.alpha = alpha

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

# Create smoothers for volume and speed.
volume_smoother = SmoothValue(alpha=0.1)
speed_smoother = SmoothValue(alpha=0.1)
# If needed, you could add a smoother for pitch as well.

# =============================
# Hand Tracking with MediaPipe
# =============================
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

    image = cv2.flip(image, 1)  # Mirror the image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_image)

    # Process only if two hands are detected.
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1 = results.multi_hand_landmarks[0]
        hand2 = results.multi_hand_landmarks[1]

        # Get landmarks for thumb and index finger for both hands.
        thumb1 = hand1.landmark[mphands.HandLandmark.THUMB_TIP]
        index1 = hand1.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
        thumb2 = hand2.landmark[mphands.HandLandmark.THUMB_TIP]
        index2 = hand2.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]

        h, w, _ = image.shape
        thumb1x, thumb1y = int(thumb1.x * w), int(thumb1.y * h)
        index1x, index1y = int(index1.x * w), int(index1.y * h)
        thumb2x, thumb2y = int(thumb2.x * w), int(thumb2.y * h)
        index2x, index2y = int(index2.x * w), int(index2.y * h)

        # Calculate midpoints for each hand.
        mid1x = (thumb1x + index1x) // 2
        mid1y = (thumb1y + index1y) // 2
        mid2x = (thumb2x + index2x) // 2
        mid2y = (thumb2y + index2y) // 2

        # Draw circles for visualization.
        cv2.circle(image, (thumb1x, thumb1y), 10, (0, 255, 0), -1)  # Thumb in green
        cv2.circle(image, (index1x, index1y), 10, (0, 0, 255), -1)  # Index in red
        cv2.circle(image, (thumb2x, thumb2y), 10, (0, 255, 0), -1)
        cv2.circle(image, (index2x, index2y), 10, (0, 0, 255), -1)

        # Draw lines connecting key points.
        cv2.line(image, (thumb1x, thumb1y), (index1x, index1y), (255, 255, 255), 2)
        cv2.line(image, (thumb2x, thumb2y), (index2x, index2y), (255, 255, 255), 2)
        cv2.line(image, (mid1x, mid1y), (mid2x, mid2y), (255, 255, 0), 2) 

        # =============================
        # Calculate Distances & Update Audio Parameters
        # =============================
        # Calculate distances (scaled down by dividing by 10).
        dist1 = math.sqrt((thumb1x - index1x) ** 2 + (thumb1y - index1y) ** 2) / 10
        mid_dist = math.sqrt((mid2x - mid1x) ** 2 + (mid2y - mid1y) ** 2) / 10

        # For volume: map distance (dist1) to a 0.0 - 1.0 range.
        raw_volume = dist1 / 18  # Adjust the divisor as needed for your setup.
        smoothed_volume = volume_smoother.update(raw_volume)
        # Optional thresholding: only update if the change is significant.
        # if abs(smoothed_volume - sf.mul) > 0.05:
        set_volume(smoothed_volume)

        # For speed: map the distance between midpoints to a reasonable speed range.
        raw_speed = mid_dist / 20  # Adjust scaling as necessary.
        smoothed_speed = speed_smoother.update(raw_speed)
        # Clamp the speed between 0.5x and 2.0x for natural playback.
        set_speed(max(0.5, min(smoothed_speed, 2.0)))

        # Optionally, you could calculate and set pitch based on another measurement.
        # raw_pitch = some_function_of(distance)
        # smoothed_pitch = pitch_smoother.update(raw_pitch)
        # set_pitch(smoothed_pitch)

        # Display distances for visual feedback.
        cv2.putText(image, f"{int(dist1)}", ((thumb1x + index1x) // 2, (thumb1y + index1y) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"{int(mid_dist)}", ((mid1x + mid2x) // 2, (mid1y + mid2y) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Hand Tracker", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
