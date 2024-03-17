import os
import pandas as pd
import json
from dotenv import load_dotenv
from tqdm import tqdm
from paragrapher import makeParagraphs
from anonymizer import anonymize
from summarizer import getSummaries 
from questionanswerer import getQuestions

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

def main(input_filename: str, output_name: str, chains: bool, number_of_questions: int, indexes=None):
    
    # Step 1: Make Paragraphs
    paragraphs = makeParagraphs(input_filename)
    
    # Step 2: Anonymize Paragraphs
    anonymized_paragraphs = anonymize(paragraphs)
    
    # Step 3: Get Summaries
    summarized_df = getSummaries(anonymized_paragraphs, chains=chains, name=output_name, indexes=indexes)
    
    # Step 4: Generate Questions and Answers
    getQuestions(summarized_df, number_of_questions=number_of_questions, name=output_name)
    
    print("All steps completed successfully! ✅")

if __name__ == "__main__":
    while True:
        input_filename = input("Enter input filename (without .txt extension): ")        
        if os.path.exists(f"../Resources/{input_filename}.txt"):
            break
        else:
            print(f"File {input_filename} not found! Please enter a valid file name.")
    output_name = input("Enter output name: ")
    index_choice = input("Do you want to process specific indexes, sample randomly, or process all items? (enter 'specific', 'sample', or 'all'): ").strip().lower()
    if index_choice == 'specific':
        indexes_input = input("Enter specific indexes separated by commas (e.g., 0,2,4): ")
        indexes = [int(i) for i in indexes_input.split(",")]
    elif index_choice == 'sample':
        sample_size = int(input("How many items do you want to randomly sample? "))
        indexes = sample_size
    else:  # 'all'
        indexes = None
    chains_input = input("Do you want to enable chains? (yes/no): ").strip().lower()
    chains = chains_input == 'yes'
    
    number_of_questions = int(input("Enter the number of questions: "))
    
    main(input_filename, output_name, chains, number_of_questions, indexes)
