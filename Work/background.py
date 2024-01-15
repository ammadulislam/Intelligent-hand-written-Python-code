import cv2
import numpy as np

def change_background_and_text_color(input_image_path, output_image_path):
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
        image[binary == 0] = [0, 0, 0]  #

        # Save the processed image
        cv2.imwrite(output_image_path, image)

        print("Image processed successfully and saved as", output_image_path)

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
    input_image_path = 'C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\path_to_upload_folder\\118\\image_1.jpg'  # Replace with your input image file path
    output_image_path = 'output_image1.jpg'  # Output file path for the processed image

    change_background_and_text_color(input_image_path, output_image_path)
