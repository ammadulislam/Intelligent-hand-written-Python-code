import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from PIL import Image

class StrokeObject:
    def __init__(self, x, y, thickness, color):
        self.x = x
        self.y = y
        self.thickness= thickness
        self.color = color

class CharacterRecognition:
    def __init__(self, model_path):
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print("Error loading the model:", str(e))

        self.image = None  # Initialize the 'image' attribute

    def predict_character_index(self, char_image):
        # Resize the character image to match the model input size
        char_image = cv2.resize(char_image, (28, 28))

        # Preprocess the image for the model
        char_image = self.preprocess_image(char_image)

        # Make predictions using the loaded model
        predictions = self.model.predict(char_image)

        # Get the index with the highest probability as the predicted character index
        predicted_char_index = np.argmax(predictions)

        return predicted_char_index

    # def save_cropped_images(self, output_directory, my_array, bounding_boxes):
    #     for i, (x_orig, y_orig, w, h) in enumerate(bounding_boxes):
    #         # Choose the first scaled coordinate for each bounding box
    #         x_scaled, y_scaled = my_array[i]
    #
    #         # Crop the character from the grayscale image
    #         char_image = self.image[y_orig:y_orig + h, x_orig:x_orig + w]
    #
    #         # Use the original and scaled coordinates for saving
    #         char_output_filename = os.path.join(output_directory,
    #         #f'char_{i}_x{x_orig}_{x_scaled}_y{y_orig}_{y_scaled}.png')
    #         f'char_{i}_x{x_orig}_y{y_orig}.png')
    #         cv2.imwrite(char_output_filename, char_image)
    #
    #         print(f"Cropped image saved as {char_output_filename}")
    @staticmethod
    def get_image_dimensions(image_path):
        with PILImage.open(image_path) as img:
            width, height = img.size
        return width, height

    def save_to_pdf(self, output_pdf_path, char_images_with_coordinates, input_image_path,slidenum,slide_count):
        img_width, img_height = self.get_image_dimensions(input_image_path)

        pdf = canvas.Canvas(output_pdf_path, pagesize=(img_width, img_height))

        colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Add more colors as needed

        for ((char_output_filename, (x, y, w, h)), predicted_char_index) in char_images_with_coordinates:

            # Determine font size based on character height
            if 110 <= h < 120:
                font_size = 60
            elif 120 <= h < 130:
                font_size = 62
            elif 130 <= h < 140:
                font_size = 68
            elif 140 <= h < 150:
                font_size = 72
            elif 150 <= h < 160:
                font_size = 78
            elif 160 <= h < 170:
                font_size = 80
            else:
                # Default font size if height is out of specified ranges
                font_size = 16

            # Get the color based on the predicted character index
            color = colors[predicted_char_index % len(colors)]

            # Draw the character label on the PDF at the specified coordinates with the determined font size and color
            pdf.setFillColorRGB(*color)
            pdf.setFont("Helvetica", font_size)
            pdf.drawString(x, img_height - y - h - 10, f"{predicted_char_index}")

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

    def preprocess_image_and_save(self, char_image, output_path, x, y):
        # Resize the image to 28x28
        img = PILImage.fromarray(char_image)
        img = img.resize((28, 28))

        # Save the character image to the output directory with modified coordinates
        img.save(output_path)

        # Return the path to the saved character image
        return output_path

    def display_image_with_boxes(self, image, bounding_boxes):
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"Height: {h}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Processed Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def is_point_inside_rectangle(self, point, rectangle_origin, rectangle_width, rectangle_height):
        x, y = point
        x_rect, y_rect = rectangle_origin
        rect_right = x_rect + rectangle_width
        rect_bottom = y_rect + rectangle_height

        return x_rect <= x <= rect_right and y_rect <= y <= rect_bottom

    def change_background_and_text_color(self, input_image_path, output_image_path, output_directory, my_array):
        try:
            # scale the point received from API
            scalledarry = []
            for i in my_array:
                x1 = i.x
                y1 = i.y
                original_width, original_height = 342.4285714285, 730.1428571428571
                new_width, new_height = 600, 1278
                x2, y2 = scale_point(x1, y1, original_width, original_height, new_width, new_height)
                scalledarry.append((x2, y2))

            # Load the image using OpenCV
            self.image = cv2.imread(input_image_path)

            if self.image is None:
                raise Exception("Image not found.")

            # Convert the image to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

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
                self.image[y:y + h, x:x + w][roi == 255] = [255, 255, 255]  # White

            # Set the background to black
            self.image[binary == 0] = [0, 0, 0]

            # Save the processed image
            cv2.imwrite(output_image_path, self.image)

            print("Image processed successfully and saved as", output_image_path)

            # Now, process the image to extract characters with bounding boxes
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
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

            char_images_with_coordinates = []

            for i, (x, y, w, h) in enumerate(bounding_boxes):
                #print(i, x, y, w, h)  # Add this line
                # Crop the character from the grayscale image
                char_image = gray_image[y:y + h, x:x + w]

                # Find the corresponding scaled point for the current bounding box
                for j, stroke_object in enumerate(my_array):
                    #print(j, stroke_object)  # Add this line
                    #print("my_array:", my_array)
                    #print("Type of element:", type(stroke_object))
                    x_orig, y_orig = stroke_object.x, stroke_object.y
                    x_scaled, y_scaled = scalledarry[j]
                    rectangle_origin = (x, y)
                    rectangle_width = w
                    rectangle_height = h
                    point_to_check = (x_scaled, y_scaled)

                    if self.is_point_inside_rectangle(point_to_check, rectangle_origin, rectangle_width,
                                                      rectangle_height):
                        print("Point is inside the rectangle.")

                        # Use your model to predict the character index (modify this part based on your model)
                        # For now, let's assume you have a predict_character_index function
                        predicted_char_index = self.predict_character_index(char_image)

                        # Save the image with both original and scaled coordinates in the filename
                        char_output_filename = os.path.join(output_directory,
                                                            f'char_{i}_orig_x{x_orig}_y{y_orig}_scaled_x{x_scaled}_y{y_scaled}_predicted{predicted_char_index}.png')
                        print(
                            f"x_orig: {x_orig}, y_orig: {y_orig}, x_scaled: {x_scaled}, y_scaled: {y_scaled}, predicted_char_index: {predicted_char_index}")
                        cv2.imwrite(char_output_filename, char_image)

                        char_images_with_coordinates.append(
                            ((char_output_filename, (x_orig, y_orig, w, h)), predicted_char_index))
                        break

                    #Save the processed image with bounding boxes
                output_image_with_boxes_path = os.path.join(output_directory, 'output_image_with_boxes.jpg')
                # Draw bounding boxes on the image
                image_with_boxes = self.image.copy()
                for (x, y, w, h) in bounding_boxes:
                    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Save the image with bounding boxes
                cv2.imwrite(output_image_with_boxes_path, image_with_boxes)

            #Save to PDF
            output_pdf_path = os.path.join(output_directory, 'output_characters.pdf')
            self.save_to_pdf(output_pdf_path, char_images_with_coordinates,input_image_path)

            # Save cropped images using scaled coordinates
            #self.save_cropped_images(output_directory, my_array, bounding_boxes)

            print(f"Processed characters saved in '{output_directory}' directory.")
            print(f"Original image with bounding boxes saved as '{output_image_with_boxes_path}'.")
            print(f"Predicted characters and their positions saved in '{output_pdf_path}'.")


        except Exception as e:
            print("An error occurred:", str(e))



def scale_point(x, y, original_width, original_height, new_width, new_height):
    x_scaled = int((x / original_width) * new_width)
    y_scaled = int((y / original_height) * new_height)
    return x_scaled, y_scaled

if __name__ == "__main__":
    my_array = [
        StrokeObject(96, 67, "stroke1", "#38761d"),
        StrokeObject(146, 60, "stroke2", "#F55330"),
        StrokeObject(233, 60, "stroke3", "#351c75"),
        StrokeObject(73, 194, "stroke4", "#F55330"),
        StrokeObject(194, 198, "stroke5", "#F55330"),
        StrokeObject(299, 226, "stroke6", "#F55330"),
        StrokeObject(70, 353, "stroke7", "#F55330"),
    ]
   # my_array = [(96, 67), (146, 60), (233, 60), (73, 194), (194, 198), (299, 226), (70, 353)]
    #my_array = [(13, 25), (292, 43), (12, 292), (297, 295), (12, 596), (285, 600)]
    model_path = 'C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\Work\\mnist_model.h5'  # Replace with your model file path
   # input_image_path = "C:\\Users\\HP\\PycharmProjects\\Fyp_Final\\5\\image_1.jpg"
    input_image_path = "C:\\Users\\HP\\Desktop\\132.jpg"

    # Replace with your input image file path
    #input_image_path = "C:\\Users\\HP\\Desktop\\image_11.jpg"
    output_image_path = 'output_image.jpg'  # Output file path for the processed image
    output_directory = 'output'  # Output directory where processed images will be saved

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    char_recognizer = CharacterRecognition(model_path)
    char_recognizer.change_background_and_text_color(input_image_path, output_image_path, output_directory, my_array)