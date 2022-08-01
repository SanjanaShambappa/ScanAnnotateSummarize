''' 
Project Title : Extractive Summarization of Image Extracted Text
Course : Artificial Intelligence and Machine Learning (AIML) 
File Name : Luhn.py
Description :
This notebook performs extractive summarization using Luhn.  
Input : 
    The extracted text to be summarized.   
Output: 
    The summary.
'''

import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')

def run_summarization(text_string, sentence_count):
    from sumy.summarizers.luhn import LuhnSummarizer
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    language = "english"
    parser = PlaintextParser(text_string, Tokenizer(language))
    summarizer = LuhnSummarizer(Stemmer(language))
    summarizer.stop_words = get_stop_words(language)
    summary = summarizer(parser.document, sentence_count)
    return summary