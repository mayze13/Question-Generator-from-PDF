from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import math
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import re
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def rev_sigmoid(x:float)->float:
    return (1 / (1 + math.exp(0.5*x)))

def activate_similarities(similarities:np.array, p_size=10)->np.array:
        """ Function returns list of weighted sums of activated sentence similarities
        Args:
            similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
            p_size (int): number of sentences are used to calculate weighted sum 
        Returns:
            list: list of weighted sums
        """
        # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10,10,p_size)
        # Then we need to apply activation function to the created space
        y = np.vectorize(rev_sigmoid) 
        # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
        activation_weights = np.pad(y(x),(0,similarities.shape[0]-p_size))
        ### 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0,similarities.shape[0])]
        ### 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
        diagonals = [np.pad(each, (0,similarities.shape[0]-len(each))) for each in diagonals]
        ### 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)
        ### 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1,1)
        ### 5. Calculate the weighted sum of activated similarities
        activated_similarities = np.sum(diagonals, axis=0)
        return activated_similarities
    
def makeParagraphs(filename):
    """
    This function takes the filename in from pdfprompts/Resources/ and splits it into paragraphs according to local minima in vector similarity.
    
    Args:
        filename: the name that will be used to fetch the file. Exclude the .txt.
    returns:
        a list of strings that are paragraphs of the provided text.
    """
    with open(f"../Resources/{filename}.txt", 'r') as file:
        data = json.load(file)
    text = data['results']['transcripts'][0]['transcript']
    text.replace("?", ".")
    text.replace("!", ".")
    sentence_split = text.split('. ')
    print(f"There are {len(sentence_split)} sentences.")
    print("Initialising Embedding model...")
    model = SentenceTransformer('all-mpnet-base-v2')
    print("Done ✅")
    print("Trimming some outliers...")
    sentence_lengths = [len(sentence) for sentence in sentence_split]
    mean_length = np.mean(sentence_lengths)
    std_length = np.std(sentence_lengths)
    threshold = mean_length + 2 * std_length
    exceeded_indices = []
    i = 0
    while i < len(sentence_split):
        sentence = sentence_split[i]
        if len(sentence) > threshold:
            comma_index = sentence.rfind(',', 0, int(threshold))
            if comma_index != -1:
                first_part = sentence[:comma_index] + '.'
                second_part = sentence[comma_index+1:].lstrip()
                sentence_split[i:i+1] = [first_part, second_part]
                i += 1
            exceeded_indices.append(i)
        i += 1
    print("Done ✅")
    print("Encoding sentences...")
    embeddings = model.encode(sentence_split)
    print("Done ✅")
    print("Calculating local minimas...")
    similarities = cosine_similarity(embeddings)
    activated_similarities = activate_similarities(similarities, p_size=10)
    minmimas = argrelextrema(activated_similarities, np.less, order=2)
    print("Done ✅")
    print("Splitting paragraphs...")
    split_points = [each for each in minmimas[0]]
    text = ''
    for num,each in enumerate(sentence_split):
        if num in split_points:
            text+=f'\n\n {each}. '
        else:
            text+=f'{each}. '
    text_paragraph = text.split("\n\n")
    #paragraphs_dict = {i: (string, len(string)) for i, string in enumerate(text_paragraph)}
    print("All Done ✅")
    return text_paragraph