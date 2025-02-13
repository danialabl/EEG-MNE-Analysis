import numpy as np
import scipy.io
from scipy.signal import hilbert, butter, filtfilt
import matplotlib.pyplot as plt

# Load the .mat file
data = scipy.io.loadmat(r"C:\Users\ablay\Downloads\LIFU_Anvar.mat")
trials = data['LCN']['trials'][0][0][1013:1227,:,:]
n_trials, n_channels, n_timepoints = trials.shape

# Define frequency range from 1 to 30 Hz with desired resolution.
frequencies = np.linspace(1, 30, num=30)  # Example: 1 to 30 Hz with 30 points
fs = 1000  # Example sampling frequency in Hz
chnnls=["Mc","Mi","Aud"]
# Function to compute ITPC for each frequency
def compute_itpc(trials, frequencies, fs):
    n_trials, n_channels, n_timepoints = trials.shape
    n_frequencies = len(frequencies)
    itpc = np.zeros((n_channels, n_frequencies, n_timepoints), dtype=np.complex_)

    nyquist = 0.5 * fs  # Nyquist frequency

    # Loop over each frequency
    for freq_idx in range(n_frequencies):
        frequency = frequencies[freq_idx]
        # Set bandwidth of ±1 Hz
        low = max(0.01, frequency - 1) / nyquist   # Avoid 0 by using 0.01
        high = (frequency + 1) / nyquist  # Ensure this is valid

        # Proceed only if low < high
        if low >= high:
            print(f"Skipping frequency {frequency} due to invalid filter bounds: low={low}, high={high}")
            continue

        # Create a bandpass filter for the current frequency
        b, a = butter(4, [low, high], btype='band')

        # Apply the filter and compute the Hilbert transform for each channel and trial
        for trial in range(n_trials):
            for channel in range(n_channels):
                filtered_signal = filtfilt(b, a, trials[trial, channel, :])
                analytic_signal = hilbert(filtered_signal)
                phase = np.angle(analytic_signal)

                # Compute the ITPC for this channel and frequency
                itpc[channel, freq_idx, :] += np.exp(1j * phase)

    # Average across trials and normalize
    itpc = np.abs(itpc) / n_trials
    return itpc

# Compute ITPC for the specified frequency range
itpc = compute_itpc(trials, frequencies, fs)

# Plotting the heatmap for each channel across all computed frequencies
for channel in range(n_channels):
    plt.figure(figsize=(10, 6))
    plt.imshow(itpc[channel, :, :], aspect='auto', origin='lower', extent=[0, n_timepoints, frequencies[0], frequencies[-1]], cmap='jet',vmin=0, vmax=0.45)
    plt.colorbar(label='ITPC')
    plt.title(f'LCN ITPC Rat_id: 7 '+chnnls[channel])
    plt.xlabel('Time points')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()

