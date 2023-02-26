
""" Convert Stanford Car Dataset .mat file to .csv file format

usage: convert_mat2csv.py [-h] [-m MAT_DIR] [-o OUTPUT_PATH] [-i IMAGE_DIR] [--train_sample_size (int)] [--test_val_sample_size (int)] [--class_sample_size (int)]
optional arguments:
  -h, --help            show this help message and exit

  -m MAT_DIR,     --mat_dir MAT_DIR
                        Path to the folder where the input .mat files are stored.
  -o OUTPUT_PATH, --output_path OUTPUT_PATH
                        Path of output .csv file.
  -i IMAGE_DIR,   --image_dir IMAGE_DIR
                        Path to the folder where the input image files are stored. Defaults to the same directory as XML_DIR.
  -c CSV_PATH,    --csv_path CSV_PATH
                        Path of output .csv file. If none provided, then no file will be written.
                  --train_sample_size 
                        How many train data do you want to use.
                  --test_val_sample_size
                        Total number of test and validation samples
                  --class_sample_size
                        Number of classes to be used during training
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
import os
import random
import cv2
import argparse
from sklearn.model_selection import train_test_split

#################################################################       PARSER      ##################################################################################

def parse_args():

    parser = argparse.ArgumentParser(
    description="Convert Stanford Car Dataset .mat file to .csv file format")
    
    parser.add_argument("-m",
                        "--mat_dir",
                        help="Path to the folder where the input .mat files are stored.",
                        type=str)
    
    parser.add_argument("-o",
                        "--output_path",
                        help="Path of output .csv file. If none provided, then no file will be written.",
                        type=str)
    
    parser.add_argument("-i",
                        "--image_path",
                        help="Path of the folder where the images are stored",
                        type=str)
    
    parser.add_argument("--train_sample_size",
                        help="Number of train samples to be taken for each class. Default 20", 
                        type=int,
                        default = 20)
    
    parser.add_argument("--test_val_sample_size",
                        help="Number of test and validation samples to be taken for each class. Default 10",
                        type=int,
                        default=10)
    
    parser.add_argument('--class_sample_size', 
                        help="Number of classes do you want to use. Default 10",
                        type=int, 
                        default=10)

    return parser.parse_args()


#################################################################       FUNCTIONS      ##################################################################################

# Read the .mat file
def mat_to_df(annotation_path):
    
    try:
        MAT = loadmat(annotation_path)
    except:
        print("Annotations file couldn't found.")
        
    annotations = MAT["annotations"][0,:]
    nclasses = len(MAT["class_names"][0])
    class_names = dict(zip(range(1,nclasses+1),[c[0] for c in MAT["class_names"][0]]))
    
    
    dataset = []
    for arr in annotations:
        # the first entry in the row is the image name
        # The rest is the data, first bbox, then classid then a boolean for whether in train or test set
        dataset.append([arr[0][0].replace("car_ims/","")] + [str(y[0][0]) for y in arr][1:])
    # Convert to a DataFrame, and specify the column names
    df = pd.DataFrame(dataset, 
                      columns =['filename',"bbox_x1","bbox_y1","bbox_x2","bbox_y2","class_id","test"])
    
    df = df.astype({"bbox_x1": "int","bbox_y1": "int","bbox_x2": "int","bbox_y2": "int","class_id": "int","test": "int"})

    df_className_added = df.assign(class_name=df["class_id"].map(dict(class_names)))
    
    return df_className_added

"""
- Let's take samples from the dataset and work with this small dataset instead of the whole dataset.
- Let's divide the samples we received into train & val & test sets
"""

def reduce_dataset(dataframe, RANDOM_CLASS_ID, TRAIN_SAMPLE_SIZE, TEST_and_VAL_SAMPLE_SIZE):
    
    class_based_reduced_dataset = dataframe[dataframe["class_id"].isin(RANDOM_CLASS_ID)]   
    
    # Split dataset to train and test
    train_dataset = class_based_reduced_dataset[class_based_reduced_dataset["test"] == 0]
    test_dataset = class_based_reduced_dataset[class_based_reduced_dataset["test"] ==1]
    
    # Take a sample for train and test dataset
    train_dataset = train_dataset.groupby('class_id').apply(lambda x: x.sample(n=TRAIN_SAMPLE_SIZE)).reset_index(drop = True)
    test_dataset = test_dataset.groupby('class_id').apply(lambda x: x.sample(n=TEST_and_VAL_SAMPLE_SIZE)).reset_index(drop = True)
    
    test_dataset,validation_dataset = train_test_split(test_dataset, test_size = 0.5,stratify = test_dataset["class_id"])
    
    return train_dataset, test_dataset, validation_dataset

"""
- Let's prepare a final dataset.
- Read images from the dataset using filename.
"""
def prepare_final_dataset(dataset,CAR_IMAGES_PATH):
    img_height = []
    img_width = []

    # Throw error if image read fails and break the function
    try:
      cv2.imread(os.path.join(CAR_IMAGES_PATH,dataset["filename"].values[1]))
      print(f"image read successfully from {CAR_IMAGES_PATH}")
    except:
        return print(f"Something went wrong while reading the images.\nCheck the path of the file you stored.\nThe image folder path you defined \n{CAR_IMAGES_PATH}")
        
    for imgfile in dataset["filename"]:
      
      try:
          height,width,_ = cv2.imread(os.path.join(CAR_IMAGES_PATH,imgfile)).shape
          img_height.append(height)
          img_width.append(width)

      except:
          print(f"Something went wrong while reading the images.\nImage couldn't found {os.path.join(CAR_IMAGES_PATH,imgfile)}")
          continue
    
    dataset["img_height"] = img_height
    dataset["img_width"] = img_width
    dataset.drop(["test","class_id"],axis = 1,inplace = True)

    dataset = dataset.reset_index(drop=True)
    ordered_dataset = dataset[['filename', 'img_width', 'img_height', 'class_name', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']]

    final_column_names = {'filename' :'filename', 'img_width' : 'width', 'img_height' :'height', 'class_name' :'class', 'bbox_x1' :'xmin', 'bbox_y1' :'ymin', 'bbox_x2' :'xmax', 'bbox_y2' :'ymax'}

    ordered_dataset.rename(columns = final_column_names, inplace = True)
    return ordered_dataset

if __name__ == '__main__':
    
    args = parse_args()
    
    CAR_IMAGES_PATH = args.image_path
    ANNOTATION_PATH = args.mat_dir
    OUTPUT_PATH = args.output_path
    TRAIN_SAMPLE_SIZE = args.train_sample_size
    TEST_and_VAL_SAMPLE_SIZE = args.test_val_sample_size
    CLASS_SAMPLE_SIZE= args.class_sample_size
    
    RANDOM_CLASS_ID = [random.randint(1,196) for i in range(CLASS_SAMPLE_SIZE)]
    dataset = mat_to_df(ANNOTATION_PATH)
    
    train_dataset, test_dataset, validation_dataset = reduce_dataset(dataset, RANDOM_CLASS_ID, TRAIN_SAMPLE_SIZE, TEST_and_VAL_SAMPLE_SIZE)

    train_dataset = prepare_final_dataset(train_dataset,CAR_IMAGES_PATH)
    test_dataset = prepare_final_dataset(test_dataset,CAR_IMAGES_PATH)
    validation_dataset = prepare_final_dataset(validation_dataset,CAR_IMAGES_PATH)

    train_dataset.to_csv(os.path.join(OUTPUT_PATH,"train_dataset.csv"),index=False)
    test_dataset.to_csv(os.path.join(OUTPUT_PATH,"test_dataset.csv"),index=False)
    validation_dataset.to_csv(os.path.join(OUTPUT_PATH,"validation_dataset.csv"),index=False)

    if os.path.exists(os.path.join(OUTPUT_PATH,"train_dataset.csv")):
      print('Successfully created the .csv file: {}'.format(os.path.join(OUTPUT_PATH,"train_dataset.csv")))
    
    else:print("Failed to create train file")
    
    if os.path.exists(os.path.join(OUTPUT_PATH,"test_dataset.csv")):
      print('Successfully created the .csv file: {}'.format(os.path.join(OUTPUT_PATH,"test_dataset.csv")))
    else:print("Failed to create test file")

    if os.path.exists(os.path.join(OUTPUT_PATH,"validation_dataset.csv")):
      print('Successfully created the .csv file: {}'.format(os.path.join(OUTPUT_PATH,"validation_dataset.csv")))
    else:print("Failed to create validation file")




    
    
    
    
    
    
    
    
    
