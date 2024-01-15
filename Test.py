import cv2
import os
import numpy as np
import tensorflow as tf
from PIL import Image as PILImage
from reportlab.pdfgen import canvas


class StrokeObject:
    def __init__(self, x, y, stroke_name, color_code):
        self.x = x
        self.y = y
        self.stroke_name = stroke_name
        self.color_code = color_code
class CharacterRecognition:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def save_to_pdf(self, output_pdf_path, char_images_with_coordinates):
        custom_width = 600
        custom_height = 1278

        pdf = canvas.Canvas(output_pdf_path, pagesize=(custom_width, custom_height))

        for (char_output_filename, (x, y, w, h), predicted_char_index, char_size, stroke, color) in char_images_with_coordinates:
            # Determine font size based on character height
            if 110 <= h < 120:
                font_size = 26
            elif 120 <= h < 130:
                font_size = 28
            elif 130 <= h < 140:
                font_size = 30
            elif 140 <= h < 150:
                font_size = 32
            elif 150 <= h < 160:
                font_size = 34
            elif 160 <= h < 170:
                font_size = 36
            else:
                # Default font size if height is out of specified ranges
                font_size = 16

            # Draw the character label on the PDF at the specified coordinates with the determined font size
            pdf.setFont("Helvetica", font_size)
            hex_color = color.lstrip('#')
            color = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
            r,g,b = color
            # Get the color based on the predicted character index
            # Draw the character label on the PDF at the specified coordinates with the determined font size and color
            pdf.setFillColorRGB(r/255,g/255,b/255)
            pdf.setFont("Helvetica", font_size)
            #pdf.drawString(x, custom_height - y - h - 10, f"{predicted_char_index}")
            # Invert y-coordinate to match ReportLab's coordinate system
            inverted_y = custom_height - (y + h + 10)
            pdf.drawString(x, inverted_y, f"{predicted_char_index}")

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
        # Inside the loop where you process and save character images
        char_output_filename = self.preprocess_image_and_save(char_image, output_directory, x, y,target_size=(10, 10))

        img.save(char_output_filename)

        # Return the path to the saved character image
        return char_output_filename

    def display_image_with_boxes(self, image, bounding_boxes):
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"Height: {h}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Processed Image with Bounding Boxes', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def is_point_inside_rectangle(self,point, rectangle_origin, rectangle_width, rectangle_height):
        x, y = point
        x_rect, y_rect = rectangle_origin
        rect_right = x_rect + rectangle_width
        rect_bottom = y_rect + rectangle_height

        return x_rect <= x <= rect_right and y_rect <= y <= rect_bottom
    def change_background_and_text_color(self, input_image_path, output_image_path, output_directory, origional_arr):
        try:


            # scale the point recieved from api
            scalledarry = []

            for i in origional_arr:
                # x, y, s, c = i
                x1 = i.x
                y1 = i.y
                original_width, original_height = 342.4285714285, 730.1428571428571
                new_width, new_height = 600, 1278

                x2, y2 = scale_point(x1, y1, original_width, original_height, new_width, new_height)

                scalledarry.append(StrokeObject(x2, y2, i.stroke_name, i.color_code))

            #print(scalledarry)




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

            # Save the processed image
            cv2.imwrite(output_image_path, image)

            print("Image processed successfully and saved as", output_image_path)

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

            char_images_with_coordinates = []

            # Inside the loop where you process and save character images
            for i, box in enumerate(bounding_boxes):
                # Unpack bounding box values
                x, y, w, h = box

                # Crop the character from the grayscale image
                char_image = gray_image[y:y + h, x:x + w]

                # Measure the height and width of the character
                char_height = h
                char_width = w

                #pos = StrokeObject(x,y,1, "#FF5733")
                for s_point in scalledarry:
                    #
                    # Example usage

                    rectangle_origin = (x, y)
                    rectangle_width = w
                    rectangle_height = h
                    p_check_x = s_point.x
                    p_check_y = s_point.y
                    point_to_check = (p_check_x, p_check_y)

                    if self.is_point_inside_rectangle(point_to_check, rectangle_origin, rectangle_width,
                                                      rectangle_height):
                        print("Point is inside the rectangle.")
                        pos = s_point
                        # Save the character image to the output directory with bounding box coordinates
                        char_output_filename = os.path.join(output_directory, f'char_{i}_x{p_check_x}_y{p_check_y}.png')
                        cv2.imwrite(char_output_filename, char_image)

                        # Preprocess the character image and make a prediction using the model
                        processed_image = self.preprocess_image(char_image)
                        prediction = self.model.predict(processed_image)
                        predicted_class_index = np.argmax(prediction)

                        # Add additional information to the list for PDF generation
                        char_info = (
                            char_output_filename, (x, y, w, h), predicted_class_index, char_height, pos.stroke_name,
                            pos.color_code
                        )
                        char_images_with_coordinates.append(char_info)

                        break

                # Save the character image to the output directory with bounding box coordinates
                char_output_filename = os.path.join(output_directory, f'char_{i}_x{x}_y{y}.png')
                cv2.imwrite(char_output_filename, char_image)
                # Preprocess the character image and make a prediction using the model
                processed_image = self.preprocess_image(char_image)
                prediction = self.model.predict(processed_image)
                predicted_class_index = np.argmax(prediction)

                # Draw a bounding box around the character on the original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw the character label on the image with both height and width information
                label = f"Height: {char_height}, Width: {char_width}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv2.LINE_AA)

                # Append the character image and its coordinates to the list
                #char_images_with_coordinates.append(((char_output_filename, (x, y, w, h)), predicted_class_index))
            # Save the original image with bounding boxes
            output_image_with_boxes_path = os.path.join(output_directory, 'output_image_with_boxes.jpg')
            cv2.imwrite(output_image_with_boxes_path, image)



            # Save to PDF
            output_pdf_path = os.path.join(output_directory, 'output_characters.pdf')
            self.save_to_pdf(output_pdf_path, char_images_with_coordinates)

            print(f"Processed characters saved in '{output_directory}' directory.")
            print(f"Original image with bounding boxes saved as '{output_image_with_boxes_path}'.")
            print(f"Predicted characters and their positions saved in '{output_pdf_path}'.")

        except Exception as e:
            print("An error occurred:", str(e))

def scale_point(x1, y1, original_width, original_height, new_width, new_height):
    x2 = int((x1 / original_width) * new_width)
    y2 = int((y1 / original_height) * new_height)
    return x2, y2

# def Scale():
#     # Creating a list of tuples
#     my_array = [(96, 67), (146, 60), (233, 60), (73, 194), (194, 198), (299, 226), (70, 353)]
#
#     # Printing the created list
#     print(my_array)

    scalledarry = []

    for i in my_array:
        x, y = i
        x1, y1 = i
        original_width, original_height = 342.4285714285, 730.1428571428571
        new_width, new_height = 600, 1278

        x2, y2 = scale_point(x1, y1, original_width, original_height, new_width, new_height)

        scalledarry.append((x2, y2))

    print(scalledarry)


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
    # my_array = [
    #     (96, 67, "stroke1", "#FF5733"),  # Example stroke and color values
    #     (146, 60, "stroke2", "#00FF99"),
    #     (233, 60, "stroke3", "#3366CC"),
    #     (73, 194, "stroke4", "#FF5733"),
    #     (194, 198, "stroke5", "#00FF99"),
    #     (299, 226, "stroke6", "#3366CC"),
    #     (70, 353, "stroke7", "#FF5733"),
    # ]
    model_path = r'C:\Users\HP\PycharmProjects\Fyp_Final\trainedModel\data_training_example.h5'  # Replace with your model file path
    input_image_path = r"C:\Users\HP\Desktop\image_5.jpg" # Replace with your input image file path
    output_image_path = 'output_image.jpg'  # Output file path for the processed image
    output_directory = 'output'  # Output directory where processed images will be saved

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    char_recognizer = CharacterRecognition(model_path)
    char_recognizer.change_background_and_text_color(input_image_path, output_image_path, output_directory, my_array)






