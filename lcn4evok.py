import numpy as np  # Import NumPy for numerical operations
import scipy.io as sio  # Import SciPy's io module for reading .mat files
import mne  # Import MNE for processing EEG data
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
from mne.time_frequency import tfr_morlet

# Define the file path for the .mat file containing the EEG data
file_path = r"C:\Users\ablay\Downloads\LIFU_Anvar.mat"

# Load the .mat file data into a dictionary
mat_data = sio.loadmat(file_path)

# Extract EEG trials and compute the mean across trials
eeg_data = mat_data['LCN']['trials'][0][0][610:796,:,:].mean(axis=0)

# Define the sampling rate (in Hz)
sampling_rate = 1000

# Determine the number of channels and time points from the EEG data shape
n_channels = eeg_data.shape[0]
n_times = eeg_data.shape[1]

# Create an MNE Info object with channel names, sampling frequency, and channel types
info = mne.create_info(ch_names=['Mc', 'Mi', 'Aud'], sfreq=sampling_rate, ch_types='eeg')

# Create a RawArray object from the EEG data and info
raw = mne.io.RawArray(eeg_data, info)

# Define the frequency range for time-frequency analysis
frequencies = np.arange(6, 30, 1)  # Frequencies from 6 to 30 Hz
n_cycles = frequencies / 2.0  # Number of cycles for each frequency (half of the frequency)

# Perform multitaper time-frequency representation on the raw EEG data
power = tfr_morlet(raw, freqs=frequencies, n_cycles=n_cycles, return_itc=False, average=True, picks="all")

# Define the baseline period: time points from 0 to 500 ms
baseline_start = 0  # start time in ms
baseline_end = 500  # end time in ms
baseline_indices = np.where((power.times * 1000 >= baseline_start) & (power.times * 1000 <= baseline_end))[0]

# Calculate the baseline mean across the specified time points (0-500 ms)
baseline_mean = np.mean(power.data[:, :, baseline_indices], axis=2, keepdims=True)  # shape: (n_channels, n_frequencies, 1)

# Subtract the baseline mean from the power data
power.data -= baseline_mean  # Broadcasting will work as expected

# Define the starting time for the plot adjustments
negative_start = -0.5

# Adjust the time values of the power data for plotting
adjusted_times = power.times + negative_start

# List of channel names for labeling the plots
chnnls = ["Mc", "Mi", "Aud"]

# Loop through each channel's power data to create individual plots
for i in range(power.data.shape[0]):
    plt.figure()  # Create a new figure for each channel
    plt.imshow(power.data[i], aspect='auto', origin='lower',
               extent=[adjusted_times[0], adjusted_times[-1], frequencies[0], frequencies[-1]],
               cmap='jet', interpolation='bilinear', vmin=np.min(power.data), vmax=np.max(power.data))  # Plot the power data as an image
    plt.colorbar(label='Power (dB)')  # Add a color bar indicating power in decibels
    plt.title(f'LCN Power Spectrogram Rat_id:5 ' + chnnls[i])  # Set the title for the plot
    plt.xlabel('Time (s)')  # Label the x-axis
    plt.ylabel('Frequency (Hz)')  # Label the y-axis
    plt.xlim([-0.5, 2.5])  # Set x-axis limits
    plt.ylim([5, 30])  # Set y-axis limits
    plt.show()  # Display the current figure