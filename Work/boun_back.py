import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

class CharacterRecognition:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def save_to_pdf(self, output_pdf_path, char_images_with_coordinates):
        pdf = canvas.Canvas(output_pdf_path, pagesize=letter)

        for char_image, (x, y) in char_images_with_coordinates:
            pdf.drawInlineImage(char_image, x, y)

        pdf.save()

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

            # Save the processed image
            cv2.imwrite(output_image_path, image)

            print("Image processed successfully and saved as", output_image_path)

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


            bounding_boxes.sort(key=lambda x: (x[1], x[0]), reverse=True)

            predicted_characters = []
            char_images_with_coordinates = []

            for i, (x, y, w, h) in enumerate(bounding_boxes):
                # Draw a bounding box around the character on the original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Crop the character from the grayscale image
                char_image = gray_image[y:y + h, x:x + w]

                # Save the character image to the output directory with bounding box coordinates
                char_output_filename = os.path.join(output_directory, f'char_{i}_x{x}_y{y}.png')
                cv2.imwrite(char_output_filename, char_image)

                # Append the character image and its coordinates to the list
                char_images_with_coordinates.append((char_output_filename, (x, y)))

                # Preprocess the character image and make a prediction using the model
                processed_image = self.preprocess_image(char_image)
                prediction = self.model.predict(processed_image)
                predicted_class_index = np.argmax(prediction)

                # Append the predicted character to the list
                predicted_characters.append(predicted_class_index)

            # Save the original image with bounding boxes
            output_image_with_boxes_path = os.path.join(output_directory, 'output_image_with_boxes.jpg')
            cv2.imwrite(output_image_with_boxes_path, image)

            # Save to PDF
            output_pdf_path = os.path.join(output_directory, 'output_characters.pdf')
            self.save_to_pdf(output_pdf_path, char_images_with_coordinates)

            print(f"Processed characters saved in '{output_directory}' directory.")
            print(f"Original image with bounding boxes saved as '{output_image_with_boxes_path}'.")
            print(f"Predicted characters and their positions saved in '{output_pdf_path}'.")
            print("Predicted characters:", predicted_characters)

        except Exception as e:
            print("An error occurred:", str(e))

if __name__ == "__main__":
    model_path = 'C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Work\\mnist_model.h5'  # Replace with your model file path
    input_image_path = r"C:\Users\HP\PycharmProjects\Fyp_Final\Data\1304\ProjectData\ss_Lecture_1_2_image_1.jpg"  # Replace with your input image file path
    output_image_path = 'output_image.jpg'  # Output file path for the processed image
    output_directory = 'output'  # Output directory where processed images will be saved

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    char_recognizer = CharacterRecognition(model_path)
    char_recognizer.change_background_and_text_color(input_image_path, output_image_path, output_directory)
