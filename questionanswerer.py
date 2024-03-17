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
from tqdm import tqdm

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
def getQuestions(dataframe, number_of_questions=2, name='output'):
    """
    This generates two questions and answers per text input.
    Args
        dataframe: output of summarizer.py.
        texts: Résumé column of the dataframe
        number of questions: the number of QA pairs to be generated per text input.
        name: name of output csv file
    returns
        df2 but with added q&a columns
    """   
    chat = ChatOpenAI(temperature=0.3, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    response_schemas = [
        ResponseSchema(name="prompt", description="question générée sur le texte"),
        ResponseSchema(name="completion", description="réponse à la question générée")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt_pattern = r'"prompt": "(.*?)"'
    completion_pattern = r'"completion": "(.*?)"'
    
    template = """
    Tu es un coach d'exception. 
    Génère {nq} questions en français basées sur ce texte.
    Pour chaque question, génère une réponse très détaillée.
    {text}

    {format_instructions}
    """
    prompt = ChatPromptTemplate.from_template(template=template)
    dataframe[f'Q1'] = ""
    dataframe[f'A1'] = ""
    dataframe[f'Q2'] = ""
    dataframe[f'A2'] = ""

    for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc='Generating Questions and Answers...'):
        summary = row['Résumé']
        messages = prompt.format_messages(nq=number_of_questions, text=summary, format_instructions=format_instructions)
        response = chat(messages)
        data = response.content
        prompts = re.findall(prompt_pattern, data, re.DOTALL)
        completions = re.findall(completion_pattern, data, re.DOTALL)
        dataframe.at[index, f'Q1'] = prompts[0] if len(prompts) > 0 else ""
        dataframe.at[index, f'A1'] = completions[0] if len(completions) > 0 else ""
        dataframe.at[index, f'Q2'] = prompts[1] if len(prompts) > 1 else ""
        dataframe.at[index, f'A2'] = completions[1] if len(completions) > 1 else ""
    dataframe.to_csv(f'../TranscriptQAOutputs/QA/{name}_QA.csv', index=False)
    print("Done ✅")
    print(f"{name} successfully saved to pdfprompts/TranscriptQAOutputs/QA/{name}_QA.csv ✅")

