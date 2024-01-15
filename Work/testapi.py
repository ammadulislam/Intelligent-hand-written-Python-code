from flask import Flask, request, jsonify
import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage

app = Flask(__name__)

class CharacterRecognition:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def change_background_and_text_color(self, input_image_path, output_image_path, output_directory):
        try:
            # Load the image using OpenCV
            image = cv2.imread(input_image_path)

            if image is None:
                raise Exception("Image not found.")

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply adaptive thresholding to segment characters
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Use morphological operations to remove small noise
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Find contours in the binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Set the text regions to white
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                roi = binary[y:y+h, x:x+w]
                image[y:y+h, x:x+w][roi == 255] = [255, 255, 255]  # White

            # Set the background to black
            image[binary == 0] = [0, 0, 0]

            # Now, process the image to extract characters with bounding boxes
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = []

            for contour in contours:
                if cv2.contourArea(contour) > 50:  # Adjust the area threshold as needed
                    (x, y, w, h) = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, w, h))

            bounding_boxes.sort(key=lambda x: x[0])

            predicted_characters = []

            for i, (x, y, w, h) in enumerate(bounding_boxes):
                # Draw a bounding box around the character on the original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the character from the grayscale image
                char_image = gray_image[y:y + h, x:x + w]

                # Preprocess the character image and make a prediction using the model
                processed_image = self.preprocess_image(char_image)
                prediction = self.model.predict(processed_image)
                predicted_class_index = np.argmax(prediction)

                # Append the predicted character to the list
                predicted_characters.append(chr(65 + predicted_class_index))

            # Print the predicted characters to the console
            print("Predicted Characters:", "".join(predicted_characters))

            # Save the original image with bounding boxes
            cv2.imwrite(output_image_path, image)

        except Exception as e:
            print("An error occurred:", str(e))

    def preprocess_image(self, char_image):
        # Resize the image to 28x28
        img = PILImage.fromarray(char_image)
        img = img.resize((28, 28))

        # Create a new image with a black background
        new_img = PILImage.new('L', img.size, 'black')

        # Paste the character (white) onto the black background
        new_img.paste(img, (0, 0))

        img = np.array(new_img)
        img = np.expand_dims(img, axis=0)  # Expand along the first axis
        img = img / 255.0
        return img

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the image file from the POST request
        image_file = request.files['image']

        if image_file is None:
            return jsonify({"error": "No image provided"})



        # Process the image and make predictions
        char_recognizer = CharacterRecognition(model_path)
        output_directory = 'output'  # Output directory where processed images will be saved

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        output_image_path = os.path.join(output_directory, 'output_image_with_boxes.jpg')  # Output file path for the processed image

        char_recognizer.change_background_and_text_color(output_image_path, output_directory)

        return jsonify({"message": "Prediction completed. Check the terminal for predicted characters."})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    model_path = 'C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Work\\mnist_model.h5'  # Replace with your model file path

    app.run(debug=True)
