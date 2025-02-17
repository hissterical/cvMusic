import pyaudio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Audio Stream Settings
CHUNK = 1024  # Number of audio samples per frame
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate (Hz)

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Set up Matplotlib figure
fig, ax = plt.subplots()
x = np.arange(0, CHUNK)  # Time axis
line, = ax.plot(x, np.random.rand(CHUNK), '-', lw=1)
ax.set_ylim(-5000, 5000)  # Adjust amplitude range
ax.set_xlim(0, CHUNK)
ax.set_title("Real-Time Audio Waveform")
ax.set_xlabel("Samples")
ax.set_ylabel("Amplitude")

# Function to update waveform dynamically
def update_waveform(frame):
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    print(data)
    line.set_ydata(data)
    return line,

# Animate plot
ani = animation.FuncAnimation(fig, update_waveform, interval=30, blit=True)
plt.show()

# Stop and close the stream when done
stream.stop_stream()
stream.close()
p.terminate()
