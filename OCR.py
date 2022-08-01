''' 
Project Title : Extractive Summarization of Image Extracted Text
Course : Artificial Intelligence and Machine Learning (AIML) 
File Name : OCR.py
Description :
This notebook performs Optical Character Recognition (OCR).  
Input : 
    File path:
        The path to the dataset containing image files and corresponding xml files. 
        [The image files are annotated to highlight the textual content. The annotations are saved as an xml file].  
Output: 
    A dataframe containing the below columns, 
        1. File name : The name of the file 
        2. Annotation : The name of the bounding box (Ex: BB1, BB2...so on)
        3. Text : The extracted text from within the bounding box . 
'''

# Importing required packages
import cv2
import numpy as np
from matplotlib import pyplot as plt 
from google.colab.patches import cv2_imshow
import pandas as pd
import os
import pandas_read_xml as pdx
from pandas_read_xml import flatten, fully_flatten, auto_separate_tables
# Returns unmodified output as string from Tesseract OCR processing
from pytesseract import image_to_string

try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract

def getting_file_list(filePath):
        dataframe = pd.DataFrame()
        fnames_list = []
        # Extracting all files with the extension .xml from the path
        for filename in os.listdir(filePath):
            if filename.endswith("xml"):
              fnames_list.append(filename)
        # Flattening the tree structure of the xml document into the tabular structure of a dataframe
        for fname in fnames_list:
            path = filePath + fname
            df = pdx.read_xml(path, ['annotation'])
            df = df.pipe(flatten)
            df = df.pipe(flatten)
            dataframe = dataframe.append(df)
        return dataframe, fnames_list

def constructing_dataframe_from_xml(dataframe):
        # Selecting the required features
        dataframe = dataframe[["filename","object|name","object|bndbox"]]
        # Renaming the selected features
        dataframe.rename(columns = {'object|bndbox':'Coordinates', 'filename':'Filename', 'object|name': 'Annotation'}, inplace = True)
        dataframe2 = dataframe.copy()
        del dataframe['Coordinates']
        # Getting the list of unique filenames
        fname_lst = dataframe['Filename'].unique().tolist()
        # Expanding the json value into respective columns 
        dataframe2 = dataframe2.Coordinates.dropna().apply(pd.Series)
        # Concatinating the two dataframes
        result_df = pd.concat([dataframe, dataframe2.reindex(dataframe.index)], axis=1)
        return result_df

# Splicing the image based on the boundary box coordinates
def cropping_image_based_on_coordinates(img, xmin, xmax, ymin, ymax):
    crop_img = img[int(ymin) : int(ymax), int(xmin) : int(xmax)]
    return crop_img

# Using image_to_string to return unmodified output as string from Tesseract OCR processing
def ocr_using_pytesseract(crop_img):
  return pytesseract.image_to_string(crop_img)

def extract_text_from_image(result_df, fname_lst, filePath):
        # Creating a new dataframe to store the extracted text
        final_df = pd.DataFrame(columns=["Filename", "Annotation", "Text"])
        # Iterate through each unique file
        for fname in os.listdir(filePath):
            if fname.endswith("jpg"):
                path = filePath + fname
                img = cv2.imread(path, 0)
                xmin = result_df[result_df['Filename']==fname]['xmin'].to_numpy()
                ymin = result_df[result_df['Filename']==fname]['ymin'].to_numpy()
                xmax = result_df[result_df['Filename']==fname]['xmax'].to_numpy()
                ymax = result_df[result_df['Filename']==fname]['ymax'].to_numpy()
                label = result_df[result_df['Filename']==fname]['Annotation'].to_numpy()
                numOfRows = len(label)
                for i in range(numOfRows):
                    #cropping the image by passing the co-ordinates
                    crop_img = cropping_image_based_on_coordinates(img, xmin[i], xmax[i], ymin[i], ymax[i])
                    # print(fname + " - " + label[i])
                    # cv2_imshow(crop_img)
                    str1 = ocr_using_pytesseract(crop_img)
                    final_df = final_df.append({'Filename' : fname,'Annotation' : label[i] , 'Text' : str1}, 
                                ignore_index=True)
                    # print(text)
        return final_df