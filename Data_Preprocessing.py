''' 
Project Title : Extractive Summarization of Image Extracted Text
Course : Artificial Intelligence and Machine Learning (AIML) 
File Name : Data_Preprocessing.py
Description :
This notebook performs data preprocessing on text data. The extracted text is subjected to the below preprocessing, 
    1. Removing new line characters 
    2. Removing citations 
    3. Removing section numbers 
    4. Removing extra spaces 
    5. Removing accented characters 
    6. Removing special characters 
    7. Expanding contractions 
Input : 
    Dataframe:
        A dataframe containing the file name, annotation and extracted text.   
Output: 
    Dataframe:
        A dataframe containing the file name, annotation and preprocessed text.
'''

from bs4 import BeautifulSoup
import spacy
import unidecode
from word2number import w2n
import contractions
from nltk.tokenize.treebank import TreebankWordDetokenizer

def removing_new_line_char(row, final_df):
      text_string = final_df.iloc[row]['Text']
      # print("BEFORE:", text_string)
      text_string = text_string.replace('\n',' ')
      final_df.iloc[row]['Text'] = text_string \
      # print("AFTER:", text_string)

def removing_citation(row, final_df):
      text_string = final_df.iloc[row]['Text']
      #print("BEFORE:", text_string)
      idx = idx1 = 0
      while(idx != -1 and idx1 != -1):
          idx1 = text_string.find('[')
          idx = text_string.find(']')
          citation = text_string[idx1:idx+1]
          text_string = text_string.replace(citation, ' ', 1)
      final_df.iloc[row]['Text'] = text_string
      #print("AFTER:", text_string)

def removing_section_numbers(row, final_df):
      text_string = final_df.iloc[row]['Text']
      # print("BEFORE:", text_string)
      for ch in text_string:
          if(not ch.isalpha()):
              # print(ch)
              text_string = text_string.replace(ch, ' ', 1)
          else:
              break
      final_df.iloc[row]['Text'] = text_string 
      # print("AFTER:", text_string)

def removing_extra_spaces(row, final_df):
      import re
      text_string = final_df.iloc[row]['Text']
      # print("BEFORE:", text_string)
      text_string = text_string.strip()
      text_string = re.sub(' +', ' ', text_string)
      final_df.iloc[row]['Text'] = text_string 
      # print("AFTER:", text_string)

def removing_accented_chars(row, final_df):
      text_string = final_df.iloc[row]['Text']
      text_string = unidecode.unidecode(text_string)
      final_df.iloc[row]['Text'] = text_string

def expand_contractions(row, final_df):
      text_string = final_df.iloc[row]['Text']
      text_string = contractions.fix(text_string)
      final_df.iloc[row]['Text'] = text_string

def removing_special_chars(row, final_df):
      import re
      text_string = final_df.iloc[row]['Text']
      text_string = re.sub('[/?-]', '', text_string)
      final_df.iloc[row]['Text'] = text_string

def preprocess_text_in_dataframe(final_df):
      number_of_rows = final_df.shape[0]
      for i in range(number_of_rows):
            removing_new_line_char(i, final_df)
            removing_citation(i, final_df)
            removing_section_numbers(i, final_df)
            removing_accented_chars(i, final_df)
            expand_contractions(i, final_df)
            removing_special_chars(i, final_df)
            removing_extra_spaces(i, final_df)
      return final_df