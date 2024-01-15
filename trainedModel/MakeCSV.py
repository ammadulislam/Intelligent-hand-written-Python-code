import os
import csv

# Define the file path and labels
filepath = 'dataset.csv'
labels = ['path', 'label']
path='D:\\BIIT\\DataSet'
dir=os.listdir(path)
c=0
for i in dir:
    dir[c]=path+i
    c+=1
# Open the CSV file and write the headers
with open('D:\\BIIT\\DataSet\\LabelData.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(labels)

    # Iterate over each subdirectory (a, b, c) in the main directory and write the path and label to the CSV
    for subdir in os.listdir('D:\\BIIT\\DataSet'):
        subdir_path = os.path.join('D:\\BIIT\\DataSet', subdir)
        for filename in os.listdir(subdir_path):
            writer.writerow([os.path.join(subdir_path, filename), subdir])
