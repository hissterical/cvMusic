from pyo import *
import cv2
import mediapipe as mp
import math
import numpy as np

# --- Pyo Audio Setup ---
s = Server().boot()
s.start()

sf = SfPlayer("heads.wav", speed=1, loop=True, mul=0.5)
harm = Harmonizer(sf, transpo=0).out()
freq_shift = FreqShift(sf, shift=0).out()

def set_volume(volume: float):
    sf.mul = max(0.0, min(volume, 1.0))

def set_speed(speed: float):
    sf.speed = max(0.1, speed)

def set_pitch(semitones: float):
    harm.transpo = semitones

def set_frequency_shift(shift_hz: float):
    freq_shift.shift = shift_hz

# A helper class to smooth out rapid changes.
class SmoothValue:
    def __init__(self, alpha=0.1):
        self.value = None
        self.alpha = alpha

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

volume_smoother = SmoothValue(alpha=0.1)
speed_smoother = SmoothValue(alpha=0.1)

# Create a table to capture a short snippet (0.1 sec) of audio.
table = NewTable(length=0.1)
rec = TableRec(sf, table, fadetime=0.01)
rec.play()  # Start the initial recording

# Function to "clear" the table and restart recording.
def restart_rec():
    # Replace the table's contents with zeros.
    table.replace([0.0] * table.getSize())
    rec.play()

# Use a Pattern to call restart_rec every 0.1 seconds.
pat = Pattern(restart_rec, time=0.1).play()

# --- Mediapipe Hand Tracking Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands
cap = cv2.VideoCapture(0)
hands = mphands.Hands()


mid_dist=1
while True:
    success, image = cap.read()
    if not success:
        continue

    # Mirror the image.
    image = cv2.flip(image, 1)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        hand1 = None
        hand2 = None
        handedness_info = results.multi_handedness
        hand_landmarks = results.multi_hand_landmarks

        # Assign hand1 as left and hand2 as right
        if handedness_info[0].classification[0].label == 'Left' and handedness_info[1].classification[0].label == 'Right':
            hand1 = hand_landmarks[0]
            hand2 = hand_landmarks[1]
        elif handedness_info[0].classification[0].label == 'Right' and handedness_info[1].classification[0].label == 'Left':
            hand1 = hand_landmarks[1]
            hand2 = hand_landmarks[0]
        else: # Fallback in case of unsure sorting (though should not happen often)
            hand1 = hand_landmarks[0]
            hand2 = hand_landmarks[1]

        thumb1 = hand1.landmark[mphands.HandLandmark.THUMB_TIP]
        index1 = hand1.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
        thumb2 = hand2.landmark[mphands.HandLandmark.THUMB_TIP]
        index2 = hand2.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]

        h, w, _ = image.shape
        thumb1x, thumb1y = int(thumb1.x * w), int(thumb1.y * h)
        index1x, index1y = int(index1.x * w), int(index1.y * h)
        thumb2x, thumb2y = int(thumb2.x * w), int(thumb2.y * h)
        index2x, index2y = int(index2.x * w), int(index2.y * h)

        mid1x = (thumb1x + index1x) // 2
        mid1y = (thumb1y + index1y) // 2
        mid2x = (thumb2x + index2x) // 2
        mid2y = (thumb2y + index2y) // 2

        cv2.circle(image, (thumb1x, thumb1y), 10, (0, 255, 0), -1)
        cv2.circle(image, (index1x, index1y), 10, (0, 0, 255), -1)
        cv2.circle(image, (thumb2x, thumb2y), 10, (0, 255, 0), -1)
        cv2.circle(image, (index2x, index2y), 10, (0, 0, 255), -1)

        cv2.line(image, (thumb1x, thumb1y), (index1x, index1y), (255, 255, 255), 2)
        cv2.line(image, (thumb2x, thumb2y), (index2x, index2y), (255, 255, 255), 2)
        cv2.line(image, (mid1x, mid1y), (mid2x, mid2y), (255, 255, 0), 2)

        dist1 = math.sqrt((thumb1x - index1x)**2 + (thumb1y - index1y)**2) / 10
        mid_dist = math.sqrt((mid2x - mid1x)**2 + (mid2y - mid1y)**2) / 10

        raw_volume = dist1 / 18
        smoothed_volume = volume_smoother.update(raw_volume)
        set_volume(smoothed_volume)

        raw_speed = mid_dist / 20
        smoothed_speed = speed_smoother.update(raw_speed)
        set_speed(max(0.5, min(smoothed_speed, 2.0)))

        cv2.putText(image, f"{int(dist1)}", ((thumb1x + index1x) // 2, (thumb1y + index1y) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"{int(mid_dist)}", ((mid1x + mid2x) // 2, (mid1y + mid2y) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Audio Spectrum Analysis ---
    samples = table.getBuffer()
    fft_result = np.fft.rfft(samples)
    fft_magnitude = np.abs(fft_result)
    max_val = np.max(fft_magnitude) if np.max(fft_magnitude) != 0 else 1
    fft_magnitude = fft_magnitude / max_val

    num_bins = len(fft_magnitude)
    bin_width = image.shape[1] / num_bins
    for i, magnitude in enumerate(fft_magnitude):
        x = int(i * bin_width)
        bar_height = int(magnitude * 100)  # Adjust scaling factor as needed.
        cv2.rectangle(image,
                      (x, image.shape[0]),
                      (x + int(bin_width), image.shape[0] - bar_height),
                      (255, 0, 0), -1)

    cv2.imshow("Hand Tracker with Spectrum", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
