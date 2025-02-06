import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Load the .mat file
mat_file = r"C:\Users\ablay\Downloads\LIFU_Anvar.mat"   # Change this to your .mat file path
data_dict = scipy.io.loadmat(mat_file)

# Assuming the data is stored under the key 'data'
# Modify 'data' according to your file's structure
data = data_dict['AUD']['trials'][0][0][0:184,:,:] # This should be of shape (trials, channels, timepoints)

# Check the shape of the data
print(f'Data shape: {data.shape}')  # Expecting (n_trials, n_channels, n_timepoints)

# Calculate the ITPC
trials, channels, timepoints = data.shape

# Initialize the ITPC array
itpc = np.zeros((channels, timepoints))
chnnls=["Mc","Mi","Aud"]
# Loop over each channel
for ch in range(channels):
    # Extract the data for the current channel
    channel_data = data[:, ch, :]  # Shape: (trials, timepoints)

    # Calculate the analytic signal using the fast Fourier transform (FFT)
    analytic_signal = hilbert(channel_data, axis=0)

    # Get the phase of the analytic signal
    phase = np.angle(analytic_signal)

    # Compute the ITPC: Mean of the complex exponentials across trials
    complex_itpc = np.mean(np.exp(1j * phase), axis=0)

    # Compute the absolute value of ITPC
    itpc[ch, :] = np.abs(complex_itpc)

# Now you have the ITPC for each channel over time
# You can visualize the ITPC
# Example: Plot ITPC for the first channel
for i in range(3):
    plt.figure(figsize=(10, 6))
    plt.plot(itpc[i, :])
    plt.title('AUD Intertrial Phase Coherence (ITPC) for Rat_id: 2 '+chnnls[i])
    plt.xlabel('Timepoints')
    plt.ylabel('ITPC')
    plt.grid()
    
    plt.show()