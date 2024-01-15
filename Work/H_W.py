import cv2
import os
import numpy as np

class CharacterRecognition:
    def __init__(self):
        # Add any initialization code here, such as loading the model if needed.
        pass

    def find_image_corners(self, image):
        # Get image dimensions
        height, width, _ = image.shape

        # Coordinates of the corners
        top_left = (0, 0)
        top_right = (width, 0)
        bottom_left = (0, height)
        bottom_right = (width, height)

        return top_left, top_right, bottom_left, bottom_right

    def change_background_and_text_color(self, input_image_path):
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
                roi = binary[y:y + h, x:x + w]
                image[y:y + h, x:x + w][roi == 255] = [255, 255, 255]  # White

            # Set the background to black
            image[binary == 0] = [0, 0, 0]

            # Now, process the image to extract characters with bounding boxes
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                 11, 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            cleaned_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
            dilated_image = cv2.dilate(cleaned_image, None, iterations=2)

            contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            bounding_boxes = []

            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Adjust the area threshold as needed
                    (x, y, w, h) = cv2.boundingRect(contour)

                    # Filter based on aspect ratio
                    aspect_ratio = w / float(h)
                    if 0.5 <= aspect_ratio <= 2.0:  # Adjust the aspect ratio range as needed
                        bounding_boxes.append((x, y, w, h))

            bounding_boxes.sort(key=lambda x: (x[1], x[0]), reverse=True)

            # Coordinates of the corners of the full image
            top_left, top_right, bottom_left, bottom_right = self.find_image_corners(image)

            # Display the image with bounding boxes and corners
            for i, (x, y, w, h) in enumerate(bounding_boxes):
                # Draw a bounding box around the character on the original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate the coordinates of the top-right and bottom-left corners
                top_right_corner = (x + w, y)
                bottom_left_corner = (x, y + h)

                # Print the coordinates of the corners
                print(f"Character {i + 1}: Top-left ({x}, {y}), Top-right ({top_right_corner[0]}, {top_right_corner[1]}), Bottom-left ({bottom_left_corner[0]}, {bottom_left_corner[1]}), Bottom-right ({x + w}, {y + h})")

                # Calculate the area of the bounding box
                area = w * h

                # Draw the character label on the image with height, width, and area information
                label = f"Height: {h}, Width: {w}, Area: {area}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

            # Draw lines connecting the corners
            cv2.line(image, top_left, top_right, (0, 0, 255), 2)
            cv2.line(image, top_right, bottom_right, (0, 0, 255), 2)
            cv2.line(image, bottom_right, bottom_left, (0, 0, 255), 2)
            cv2.line(image, bottom_left, top_left, (0, 0, 255), 2)

            # Display the image with bounding boxes and corners
            cv2.imshow('Processed Image with Bounding Boxes and Corners', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print(f"Top Left: {top_left}, Top Right: {top_right}, Bottom Left: {bottom_left}, Bottom Right: {bottom_right}")

            # Check if a specific point is inside any bounding box
            point_to_check = (269, 386)
            point_found = False

            for i, (x, y, w, h) in enumerate(bounding_boxes):
                x1, y1, x2, y2 = x, y, x + w, y + h

                if x1 <= point_to_check[0] <= x2 and y1 <= point_to_check[1] <= y2:
                    print(f"The point {point_to_check} is inside bounding box {i + 1}.")
                    point_found = True
                    break

            if not point_found:
                print(f"The point {point_to_check} is outside all bounding boxes.")

        except Exception as e:
            print("An error occurred:", str(e))

if __name__ == "__main__":
    input_image_path = r"C:\Users\HP\PycharmProjects\Fyp_Final\Data\1502\ProjectData\output_image_with_boxes_slide_1.jpg"  # Replace with your input image file path

    char_recognizer = CharacterRecognition()
    char_recognizer.change_background_and_text_color(input_image_path)
