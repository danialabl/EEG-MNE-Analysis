import numpy as np
import mne
from scipy.io import loadmat
from mne.time_frequency import tfr_morlet
import matplotlib.pyplot as plt
from mne.time_frequency import tfr_multitaper
# Load the .mat file
mat_file_path = r"C:\Users\ablay\Downloads\LIFU_Anvar.mat"  # Change to your .mat file path
mat_data = loadmat(mat_file_path)

# Extract data
eeg_data = mat_data['AUD']['trials'][0][0][590:780,:,:]  # Adjust the indices based on your file contents
fs = 1000         # Get the sampling frequency

# Define the channel names (adjust this to reflect your dataset)
ch_names = ['Mc', 'Mi', 'Aud']
info = mne.create_info(ch_names=ch_names, sfreq=fs)

# Initialize empty list to store raw objects for each trial
raws = []

# Iterate over trials
for i in range(eeg_data.shape[0]):
    raw_trial = mne.io.RawArray(eeg_data[i, :, :], info)
    raws.append(raw_trial)

# Define frequency range and settings
freqs = np.arange(1, 30, 1)  # Frequencies of interest (e.g., 6-30 Hz)
n_cycles = freqs / 2.0          # Number of cycles per frequency

# Initialize empty list to store time-frequency representations for each trial
powers = []

# Iterate over trials
for raw_trial in raws:
    # Explicitly select the channels to use; here we assume all channels
    picks = mne.pick_channels(raw_trial.ch_names, include=ch_names)
    
    # Compute the time-frequency representation
    power_trial = tfr_morlet(raw_trial, freqs=freqs, n_cycles=n_cycles, return_itc=False, average=True, picks="all")

    powers.append(power_trial)

# Average the time-frequency representations across trials
powered_avg = np.mean([p.data for p in powers], axis=0)
baseline_period=powered_avg[:,:500]
baseline_mean=np.mean(baseline_period,axis=1,keepdims=True)
powered_avg=powered_avg-baseline_mean
for i in range(powered_avg.data.shape[0]):
    plt.figure()  # Create a new figure for each channel
    plt.imshow(powered_avg[i], aspect='auto', origin='lower',
               extent=[-0.5, 2.5, freqs[0], freqs[-1]],
               cmap='jet', interpolation='bilinear',vmin=np.min(powered_avg), vmax=np.max(powered_avg)) # Plot the power data as an image
    plt.colorbar(label='Power (dB)')  # Add a color bar indicating power in decibels
    plt.title(f'AUD Power Spectrogram Rat_id: 6 '+ch_names[i])  # Set the title for the plot
    plt.xlabel('Time (s)')  # Label the x-axis
    plt.ylabel('Frequency (Hz)')  # Label the y-axis
    plt.xlim([-0.5, 2.5])  # Set x-axis limits
    plt.ylim([1, 30])  # Set y-axis limits
    plt.show()  # Display the current figure
