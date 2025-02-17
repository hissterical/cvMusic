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
x = np.fft.rfftfreq(CHUNK, 1/RATE)  # Frequency axis
line, = ax.plot(x, (np.zeros(len(x))), '-', lw=1)
ax.set_xlim(20, RATE // 2)  # Human hearing range
ax.set_ylim(0, 10000)  # Adjust based on signal strength
ax.set_xscale("log")  # Log scale for better visualization
ax.set_title("Real-Time Frequency Spectrum")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")

# Function to update frequency spectrum
def update_spectrum(frame):
    data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
    fft_result = np.abs(np.fft.rfft(data))/5  # Apply FFT and get magnitude
    line.set_ydata(fft_result)
    return line,

# Animate plot
ani = animation.FuncAnimation(fig, update_spectrum, interval=30, blit=True)
plt.show()

# Stop and close the stream when done
stream.stop_stream()
stream.close()
p.terminate()
