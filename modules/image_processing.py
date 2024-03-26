import cv2
import numpy as np

def preprocess_image(image_path, target_size=(224, 224), grayscale=True):
    original_image = cv2.imread(image_path)

    # Convert the image to grayscale if specified
    if grayscale:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    resized_image = cv2.resize(original_image, target_size)
    normalized_image = resized_image / 255.0
    return normalized_image

def simulate_uv_vis_spectrum(image):
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 1])
    normalized_histogram = histogram / np.sum(histogram)

    wavelengths = np.arange(300, 801, 10)  # Simulated wavelengths from 300 to 800 nm

    # Generate spectrum based on image histogram
    spectrum_data = np.interp(wavelengths, np.linspace(300, 800, 256), normalized_histogram)

    return wavelengths, spectrum_data

def calculate_band_averages(wavelengths, spectrum_data, bands):
    band_averages = []
    for start, end in bands:
        indices = np.where((wavelengths >= start) & (wavelengths <= end))
        band_intensity = np.mean(spectrum_data[indices])
        band_averages.append(band_intensity)
    return band_averages

def generate_wavelength_bands():
    # Generate wavelength bands dynamically
    band_width = 10
    min_wavelength, max_wavelength = 400, 800
    bands = [(start, start + band_width) for start in range(min_wavelength, max_wavelength, band_width)]
    return bands