''' 
Project Title : Extractive Summarization of Image Extracted Text
Course : Artificial Intelligence and Machine Learning (AIML) 
File Name : T5_Model.py
Description :
This notebook performs abstractive summarization using T5 Model.  
Input : 
    The extracted text to be summarized.   
Output: 
    The summary.
'''

def init_abstractive_summarization_T5():
      import pandas as pd
      import torch
      from transformers import AutoTokenizer, AutoModelWithLMHead
      tokenizer = AutoTokenizer.from_pretrained('t5-base')
      model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict = True)  
      return tokenizer, model

def run_abstractive_summarization(text_string):
      tokenizer, model = init_abstractive_summarization_T5()
      inputs = tokenizer.encode("summarize:" + text_string, return_tensors='pt', max_length=1024, truncation = True)
      outputs = model.generate(inputs, max_length=300, min_length=80,num_beams=3)
      return tokenizer.decode(outputs[0])