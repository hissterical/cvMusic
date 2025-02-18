from pyo import *

# Initialize the audio server.
s = Server().boot()
s.start()

# Load the audio file (looping for continuous playback)
sf = SfPlayer("heads.wav", speed=1, loop=True, mul=0.5)

# Pitch shifting alternative using Harmonizer
harm = Harmonizer(sf, transpo=0).out()
freq_shift = FreqShift(sf, shift=0)  # For frequency shift

# GUI for real-time adjustments
#s.gui(locals())


# Output the processed audio
harm.out()
freq_shift.out()


# 🎚 Change Volume (0.0 to 1.0)
def set_volume(volume: float):
    sf.mul = max(0.0, min(volume, 1.0))  # Ensure it's in range 0.0 to 1.0


# ⏩ Change Speed (0.5 = half speed, 2.0 = double speed)
def set_speed(speed: float):
    sf.speed = max(0.1, speed)  # Prevent zero or negative speed


# 🎵 Change Pitch (semitones, e.g., 5 = up 5 semitones, -5 = down 5 semitones)
def set_pitch(semitones: float):
    harm.transpo = semitones  # Works like a real pitch shift


# 🔀 Shift Frequency (in Hz, e.g., 100 shifts everything up by 100 Hz)
def set_frequency_shift(shift_hz: float):
    freq_shift.shift = shift_hz


# ⏩ Seek to Position (in seconds)
def set_position(seconds: float):
    sf.pos = max(0, seconds)  # Ensure positive values


# ▶ Play Audio (if stopped)
def play():
    sf.out()


# ⏸ Pause Audio
def pause():
    sf.stop()


# 🛑 Stop Audio
def stop():
    sf.stop()
    sf.pos = 0  # Reset position


# # Example Usage
# if __name__ == "__main__":
#     set_volume(0.8)  # Set volume to 80%
#     set_speed(1.2)   # Increase speed by 20%
#     set_pitch(3)     # Shift pitch up by 3 semitones
#     set_frequency_shift(100)  # Shift frequencies up by 100 Hz







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

        dist1 = math.sqrt((thumb1x - index1x) ** 2 + (thumb1y - index1y) ** 2)//10
        dist2 = math.sqrt((thumb2x - index2x) ** 2 + (thumb2y - index2y) ** 2)//10
        mid_dist = math.sqrt((mid2x - mid1x) ** 2 + (mid2y - mid1y) ** 2)//10

        set_volume(dist1/18)
        set_speed(mid_dist/20)
        #set_pitch(dist2/100)
        

        cv2.putText(image, f"{int(dist1)}", ((thumb1x + index1x) // 2, (thumb1y + index1y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"{int(dist2)}", ((thumb2x + index2x) // 2, (thumb2y + index2y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"{int(mid_dist)}", ((mid1x + mid2x) // 2, (mid1y + mid2y) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Hand Tracker", image)  # Show BGR image
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()