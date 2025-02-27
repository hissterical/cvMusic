# Test with a simple sine wave.
from pyo import *
s = Server().boot()
s.start()
tone = Sine(freq=220, mul=0.5)
fs_tone = FreqShift(tone, shift=500)  # Apply 500 Hz shift.
fs_tone.out()
s.gui(locals())
