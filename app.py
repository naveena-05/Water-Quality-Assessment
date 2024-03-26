# app.py
from flask import Flask, render_template, request, jsonify, redirect, url_for
from modules.image_processing import preprocess_image, simulate_uv_vis_spectrum, calculate_band_averages, generate_wavelength_bands
from modules.classification import classify_sample
import os

app = Flask(__name__)

wqa_dir = os.path.join(app.root_path, 'WQA')
os.makedirs(wqa_dir, exist_ok=True)

# Initialize bands globally
bands = generate_wavelength_bands()

def handle_image_upload(image_file):
    try:
        # Save the image temporarily
        temp_path = os.path.join(wqa_dir, 'uploaded_image.png')
        image_file.save(temp_path)

        # Preprocess the image and simulate UV-Visible spectral data
        preprocessed_image = preprocess_image(temp_path, grayscale=True)
        wavelengths, uv_vis_spectrum = simulate_uv_vis_spectrum(preprocessed_image)

        # Calculate average intensity for each band
        band_averages = calculate_band_averages(wavelengths, uv_vis_spectrum, bands)
    

        # Classify the sample
        classification_result = classify_sample(band_averages)

        # Return a dictionary containing the classification result
        return {'message': 'Image processed successfully', 'classification_result': classification_result}

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return {'error': f'Error processing image: {str(e)}'}

    
@app.route('/')  # Route for the first page
def index():
    return render_template('upload_image.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'image' in request.files:
            image_file = request.files['image']
            result = handle_image_upload(image_file)

            if 'error' in result:
                return jsonify({'error': result['error']}), 500

            # Render the result page with the classification result
            return render_template('result.html', classification_result=result['classification_result'])

        return jsonify({'error': 'No image file provided'}), 400

    except Exception as e:
        return jsonify({'error': f'Error uploading image: {str(e)}'}), 500

@app.route('/result')  # Route for the second page
def result():
    classification_result = request.args.get('classification_result', 'Unknown')
    return render_template('result.html', classification_result=classification_result)

if __name__ == '__main__':
    app.run(debug=True)
