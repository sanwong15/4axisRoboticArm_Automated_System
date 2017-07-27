# Mon July 24, 2017
# Author: Yanjun Gao
#Email: yanjun.gao3@gmail.com


'''
Edit: The following code goes through data directory and create a CSV table
to record the image_name as well as its label. The CSV file will then server as
a look up table during training and testing

Update: San Wong 
Changes: (1) Automate File count in Hand/No_Hand/Idel folder
         (2) Create Train_renamed and Test_renamed csv file to store renamed Train images and Test imagss label mapping
         (3) Rename images and group them into Train image and Test image, Store spearately into Train folder and Test Folder

'''
import csv
import glob
import numpy as np
import cv2
import os


labelhand = "Hand"
labelnohand = "No_Hand"
labelidle = "Idle"
train_test_ratio = 0.7 # Subject to change: 0.7 => 70% be training

# Image Path
imageFolderPath = "dvs_every5frame_0714/"
handImg_Path = imageFolderPath + labelhand
nohandImg_Path = imageFolderPath + labelnohand
idleImg_Path = imageFolderPath + labelidle

# Count number of images in each folder
import os.path

totalnumberhand = len([f for f in os.listdir(handImg_Path)
                if os.path.isfile(os.path.join(handImg_Path, f))])

totalnumbernohand = len([f for f in os.listdir(nohandImg_Path)
                if os.path.isfile(os.path.join(nohandImg_Path, f))])

totalnumberidle = len([f for f in os.listdir(idleImg_Path)
                if os.path.isfile(os.path.join(idleImg_Path, f))])


# Create new directory to store TRAIN and TEST image

# If there doesn't exist the folder
testFolder = "Test"
testImg_Path = imageFolderPath + testFolder
if not os.path.exists(testImg_Path):
    os.makedirs(testImg_Path)

trainFolder = "Train"
trainImg_Path = imageFolderPath + trainFolder
if not os.path.exists(trainImg_Path):
    os.makedirs(trainImg_Path)

# If there's already exist the Folder. Clear all files within
# Clean Test Folder
for the_file in os.listdir(testImg_Path):
    file_path = os.path.join(testImg_Path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Expection as e:
        print(e)

# Clean Train Folder
for the_file in os.listdir(trainImg_Path):
    file_path = os.path.join(trainImg_Path, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
    except Expection as e:
        print(e)


renamed_train_header = "Train_"
renamed_test_header = "Test_"





# Create CSV file to store results: They will locate at the same directory where this code locate
# CSV file (original name of images)
with open("train.csv", "w")as f:
            writer = csv.writer(f)
            writer.writerow(["Number", "Filename", "label"])
with open("test.csv", "w")as f:
    writer = csv.writer(f)
    writer.writerow(["Number", "Filename", "label"])

# CSV file (Renamed)
with open("train_renamed.csv", "w")as f:
            writer = csv.writer(f)
            writer.writerow(["Number", "Filename", "label"])
with open("test_renamed.csv", "w")as f:
    writer = csv.writer(f)
    writer.writerow(["Number", "Filename", "label"])

i=1 # Counter for all the training csv
j=1 # Counter for all the testing csv

# Set Counter:
counter = 0

# Handle the "HAND" folder
for filename in os.listdir(handImg_Path):
    print ' Current_File: {}'.format(filename)
    print 'Counter: {}'.format(counter)

    if filename.endswith(".jpg"):
        curr_path = os.path.join(handImg_Path, filename)
        print(curr_path)
        curr_img = cv2.imread(curr_path)

        # I do think we should use While Loop that would make more sense
        if counter <= train_test_ratio * totalnumberhand:
            with open("train.csv", "a")as f:
                writer = csv.writer(f)
                writer.writerow([i, filename, labelhand])

            # Rename it
            renamed_train = renamed_train_header + str(i) + '.jpg'

            with open("train_renamed.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([i,renamed_train,labelhand])

            # Save another set of images with new name
            name_Path = trainImg_Path + '/' + renamed_train
            cv2.imwrite(str(name_Path),curr_img)
            
            #Update all counters
            i+=1
            counter+=1



        if counter > train_test_ratio * totalnumberhand:#70% of images
            with open("test.csv", "a")as f:
                writer = csv.writer(f)
                writer.writerow([j, filename, labelhand])

            renamed_test = renamed_test_header + str(j) + '.jpg'

            with open("test_renamed.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([j,renamed_test,labelhand])

            # Save another set of images with new name
            

            name_Path = os.path.join(testImg_Path,renamed_test)
            

            name_Path = testImg_Path + '/' + renamed_test
            cv2.imwrite(str(name_Path),curr_img)

            j+=1
            counter+=1
        continue
    else:
        continue



# Handle the "NO_HAND" folder
# Reset Counter
counter = 0

for filename in os.listdir(nohandImg_Path):
    print ' Current_File: {}'.format(filename)
    print 'Counter: {}'.format(counter)

    if filename.endswith(".jpg"):
        curr_path = os.path.join(nohandImg_Path, filename)
        print(curr_path)
        curr_img = cv2.imread(curr_path)

        if counter <= train_test_ratio * totalnumbernohand:
            with open("train.csv", "a")as f:
                writer = csv.writer(f)
                writer.writerow([i, filename, labelnohand])

            # Rename it
            renamed_train = renamed_train_header + str(i) + '.jpg'

            with open("train_renamed.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([i,renamed_train,labelnohand])

            # Save another set of images with new name
            name_Path = trainImg_Path + '/' + renamed_train
            cv2.imwrite(name_Path,curr_img)
            
            #Update all counters
            i+=1
            counter+=1



        if counter > train_test_ratio * totalnumbernohand:#70% of images
            with open("test.csv", "a")as f:
                writer = csv.writer(f)
                writer.writerow([j, filename, labelnohand])

            renamed_test = renamed_test_header + str(j) + '.jpg'

            with open("test_renamed.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([j,renamed_test,labelnohand])

            # Save another set of images with new name
            name_Path = testImg_Path + '/' + renamed_test
            cv2.imwrite(str(name_Path),curr_img)

            j+=1
        continue
    else:
        continue

# Handle the "Idle" folder
# Reset Counter
counter = 0

for filename in os.listdir(idleImg_Path):
    print ' Current_File: {}'.format(filename)
    print 'Counter: {}'.format(counter)

    if filename.endswith(".jpg"):
        curr_path = os.path.join(idleImg_Path, filename)
        print(curr_path)
        curr_img = cv2.imread(curr_path)

        if counter <= train_test_ratio * totalnumberidle:
            with open("train.csv", "a")as f:
                writer = csv.writer(f)
                writer.writerow([i, filename, labelidle])

            # Rename it
            renamed_train = renamed_train_header + str(i) + '.jpg'

            with open("train_renamed.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([i,renamed_train,labelidle])

            # Save another set of images with new name
            name_Path = trainImg_Path + '/' + renamed_train
            cv2.imwrite(str(name_Path),curr_img)
            
            #Update all counters
            i+=1
            counter+=1



        if counter > train_test_ratio * totalnumberidle:#70% of images
            with open("test.csv", "a")as f:
                writer = csv.writer(f)
                writer.writerow([j, filename, labelidle])

            renamed_test = renamed_test_header + str(j) + '.jpg'

            with open("test_renamed.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([j,renamed_test,labelidle])

            # Save another set of images with new name
            name_Path = testImg_Path + '/' + renamed_test
            cv2.imwrite(str(name_Path),curr_img)

            j+=1
        continue
    else:
        continue
