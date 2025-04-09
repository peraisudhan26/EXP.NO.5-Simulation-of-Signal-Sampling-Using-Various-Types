# EXP.NO.5-Simulation-of-Signal-Sampling-Using-Various-Types
5.Simulation of Signal Sampling Using Various Types such as
    i) Ideal Sampling
    ii) Natural Sampling
    iii) Flat Top Sampling

# AIM

To perform Impulse (Ideal), Natural, and Flat-Top Sampling of a continuous-time sinusoidal signal, visualize the sampled signals, and reconstruct them using Python.

# SOFTWARE REQUIRED

1. Python Software
-> Numpy Library

-> Matplotlib Library

-> Scipy Library (for signal processing)

# ALGORITHMS
i) Ideal Sampling (Impulse Sampling)

   Define the continuous-time signal 𝑥(𝑡).
   
   Set the sampling interval 𝑇𝑠.
   
   Generate an impulse train at intervals 𝑇𝑠.
   
   Multiply 𝑥(𝑡) with the impulse train to get the sampled output.

ii) Natural Sampling

Define 𝑥(𝑡) and sampling period 𝑇𝑠.

Set pulse width 𝜏(𝜏<𝑇𝑠).

Generate a periodic pulse train with width 𝜏.

Multiply 𝑥(𝑡) by the pulse train.

iii) Flat-Top Sampling

Define 𝑥(𝑡) and sampling period 𝑇𝑠.

Set pulse width 𝜏.

Generate a rectangular pulse train with width 𝜏.

Multiply 𝑥(𝑡) by the pulse train.

Hold the sampled value constant for duration 𝜏.

# PROGRAM

i) Impulse Sampling

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import resample

fs = 100

t = np.arange(0, 1, 1/fs)

f = 5

signal = np.sin(2 * np.pi * f * t)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal')

plt.title('Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

t_sampled = np.arange(0, 1, 1/fs)

signal_sampled = np.sin(2 * np.pi * f * t_sampled)

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.stem(t_sampled, signal_sampled, linefmt='r-', markerfmt='ro', basefmt='r-', label='Sampled Signal (fs = 100 Hz)')

plt.title('Sampling of Continuous Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

reconstructed_signal = resample(signal_sampled, len(t))

plt.figure(figsize=(10, 4))

plt.plot(t, signal, label='Continuous Signal', alpha=0.7)

plt.plot(t, reconstructed_signal, 'r--', label='Reconstructed Signal (fs = 100 Hz)')

plt.title('Reconstruction of Sampled Signal (fs = 100 Hz)')

plt.xlabel('Time [s]')

plt.ylabel('Amplitude')

plt.grid(True)

plt.legend()

plt.show()

ii) Natural sampling

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, lfilter

fs = 1000

T = 1 

t = np.arange(0, T, 1/fs) 

fm = 5 # Frequency of message signal (Hz)

message_signal = np.sin(2 * np.pi * fm * t)

pulse_rate = 50 

pulse_train = np.zeros_like(t)

pulse_width = int(fs / pulse_rate / 2)

for i in range(0, len(t), int(fs / pulse_rate)):

pulse_train[i:i+pulse_width] = 1

nat_signal = message_signal * pulse_train

sampled_signal = nat_signal[pulse_train == 1]

sample_times = t[pulse_train == 1]

reconstructed_signal = np.zeros_like(t)

for i, time in enumerate(sample_times):

index = np.argmin(np.abs(t - time))

reconstructed_signal[index:index+pulse_width] = sampled_signal[I]

w-pass Filter (optional, smoother reconstruction)

def lowpass_filter(signal, cutoff, fs, order=5):

nyquist = 0.5 * fs

normal_cutoff = cutoff / nyquist

b, a = butter(order, normal_cutoff, btype='low', analog=False)

return lfilter(b, a, signal)

reconstructed_signal = lowpass_filter(reconstructed_signal,10, fs)

plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)

plt.plot(t, message_signal, label='Original Message Signal')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 2)

plt.plot(t, pulse_train, label='Pulse Train')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 3)

plt.plot(t, nat_signal, label='Natural Sampling')

plt.legend()

plt.grid(True)

plt.subplot(4, 1, 4)

plt.plot(t, reconstructed_signal, label='Reconstructed Message Signal', color='green')

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()

iii) Flat top sampling

import numpy as np

import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt

fs = 1000 

t = np.arange(0, 1, 1/fs)  

f_signal = 5 

x_t = np.sin(2 * np.pi * f_signal * t)  

fs_sample = 50  

T_sample = 1/fs_sample

tau = 0.01 

t_sample = np.arange(0, 1, T_sample)

x_sample = np.sin(2 * np.pi * f_signal * t_sample)  

x_flat_top = np.zeros_like(t)

for i in range(len(t_sample)):

    idx = (t >= t_sample[i]) & (t < t_sample[i] + tau)
    
    x_flat_top[idx] = x_sample[i]

def low_pass_filter(signal, cutoff, fs):

    nyquist = fs / 2
    
    b, a = butter(5, cutoff / nyquist, btype='low')
    
    return filtfilt(b, a, signal)

x_reconstructed = low_pass_filter(x_flat_top, f_signal * 2, fs)

plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)

plt.plot(t, x_t, 'g', label="Continuous Signal")

plt.legend()

plt.subplot(3, 1, 2)

plt.plot(t, x_flat_top, 'r', label="Flat-Top Sampled Signal")

plt.stem(t_sample, x_sample, 'r', markerfmt="ro", basefmt=" ")

plt.legend()

plt.subplot(3, 1, 3)

plt.plot(t, x_reconstructed, 'b', label="Reconstructed Signal")

plt.legend()

plt.tight_layout()

plt.show()

# OUTPUT

i) Ideal Sampling:

![Screenshot 2025-03-30 121707](https://github.com/user-attachments/assets/0e8be8fe-63a9-4c01-a5c2-4db9ea9a4da0)

![Screenshot 2025-03-30 121725](https://github.com/user-attachments/assets/a835092c-331d-45b2-b4f0-6e8f13fb14e6)


ii) Natural Sampling:

![Screenshot 2025-03-30 123322](https://github.com/user-attachments/assets/806f856b-6053-4fe1-b48f-1740eaba5ebb)

![Screenshot 2025-03-30 123522](https://github.com/user-attachments/assets/81a480ba-f502-46b6-be0b-3fbf57ce2482)

iii) Flat top Sampling:

![Screenshot 2025-03-30 123926](https://github.com/user-attachments/assets/da6422e9-505f-4c0a-86ed-b8bded83c2ea)


# RESULT / CONCLUSIONS

The continuous sinusoidal signal was successfully sampled using impulse, natural, and flat-top sampling techniques. 

The sampling was performed at a rate (50 Hz) higher than twice the signal frequency (5 Hz), satisfying the Nyquist criterion. 

The reconstruction using a low-pass filter accurately recovered the original waveform, with minimal natural and slight amplitude distortion in flat-top sampling. 

The experiment demonstrates the fundamental principles of sampling and signal reconstruction.


