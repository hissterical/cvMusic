from pyo import *

# Initialize the audio server.
s = Server().boot()
s.start()

# Load the audio file (looping for continuous playback)
sf = SfPlayer("mel.wav", speed=1, loop=True, mul=0.5)

# Pitch shifting alternative using Harmonizer
harm = Harmonizer(sf, transpo=0).out()
freq_shift = FreqShift(sf, shift=0)  # For frequency shift

# GUI for real-time adjustments
s.gui(locals())


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


# Example Usage
if __name__ == "__main__":
    set_volume(0.8)  # Set volume to 80%
    set_speed(1.2)   # Increase speed by 20%
    set_pitch(3)     # Shift pitch up by 3 semitones
    set_frequency_shift(100)  # Shift frequencies up by 100 Hz

