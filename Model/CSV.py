import os
import pandas as pd
import cv2
import numpy as np

# Path to the EMNIST dataset folder (use double backslashes for Windows)
dataset_dir = 'E:\\digit'

# Function to generate a CSV file for a given dataset split (train or test)
def generate_csv(split_dir, output_csv):
    data = []
    image_count = 0
    for label, class_name in enumerate(sorted(os.listdir(split_dir))):
        class_dir = os.path.join(split_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)

            # Load the image, resize to 28x28, and normalize
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (28, 28))
            image_normalized = image / 255.0

            # Save the flattened image as a string
            flattened_image_str = ' '.join(map(str, image_normalized.flatten().tolist()))

            data.append((label, flattened_image_str))
            image_count += 1

    df = pd.DataFrame(data, columns=['label', 'image'])
    df.to_csv(output_csv, index=False)

    return image_count

# Generate CSV files for the train and test sets
train_count = generate_csv(os.path.join(dataset_dir, 'train'), 'train1.csv')
test_count = generate_csv(os.path.join(dataset_dir, 'test'), 'test1.csv')

print(f"Train CSV generated with {train_count} images.")
print(f"Test CSV generated with {test_count} images.")
