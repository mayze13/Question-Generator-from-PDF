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

def getSummaries(texts, chains=True, name='output', indexes=None):
    """
    This function makes an OpenAI call per item in the input list. 
    It fetches 5 increasinlgy complex summaries, as well as Entités Manquantes and Metadata.
    This increasingly complex summarisation based on entités manquantes is called Chain of Density.
    
    Args
        texts: a LIST of STRINGS, like the output of anonymizer.py
        
        chains: bool, return chains of density type (5 iterations, denser), else return simple summary 
        
        name: name of the output csv name of the output csv variable
    returns
        dataframe: summary, paragraph, entités manquantes, metadata
    """
    if indexes is None:
        indices_to_process = range(len(texts))
    elif isinstance(indexes, int):
        indices_to_process = np.random.choice(len(texts), indexes, replace=False)
    else:
        indices_to_process = indexes
    df = pd.read_csv('../Resources/Themes + Transcript Files IA - DECOUPAGE + IA.csv')
    filtered_df = df[(df['PEDAGOGIE'] == 'S4') & (df['Unnamed: 3'] == 'M1') & (df['Unnamed: 5'] == 'J1')]
    themes = [x for x in filtered_df['Contenu'].tolist() if not pd.isna(x)]
    themes = ', '.join(map(str, themes))
    if chains:
        template = """Article: {ARTICLE}
        Vous allez générer des résumés de plus en plus concis et riches en entités de l'article ci-dessus. Effectuez les 3 étapes suivantes 5 fois.
        Étape 1. Identifiez 1-3 Entités informatives (séparées par ";") de l'Article qui manquent dans le résumé précédemment généré.
        Étape 2. Rédigez un nouveau résumé plus dense de longueur identique qui couvre chaque entité et détail du résumé précédent plus les Entités Manquantes.
        Étape 3. Générez la metadata pour le résumé.
        Une Entité Manquante est :
        - Pertinente: par rapport à l'histoire principale.
        - Spécifique: descriptive mais concise (5 mots ou moins).
        - Nouvelle: pas dans le résumé précédent.
        - Fidèle: présente dans l'Article.
        - N'importe où: située n'importe où dans l'Article.
        La longueur de chaque résumé ne doit pas dépasser {LEN} mots. 
        La metadata doit être un dictionnaire avec les clés suivantes:
        - "themes", qui peut avoir les valeurs [{THEMES}].
        - "skills" qui peut avoir les valeurs: ["Relation", "Questionnement", "Trouver le coeur du problème", "Passage à l'action", "Communication", "Posture"]
        - "phases" qui peut avoir les valeurs: ["Etape 1 : Objectif", "Etape 2 : Réalité", "Etape 3 : Planté de drapeau", "Etape 4 : Check RED", "Etape 5 : Mindshit", "Etape 6 : Stratégie", "Etape 7 : Action", "Etape 8 : Intégration"]
        - "topics", où vous devez trouver jusqu'à 3 sujets liés au résumé.
        - "mode" qui peut avoir les valeurs: ["Pratique", "Théorique", "Les deux"]
        - "type" qui peut avoir les valeurs: ["Life coaching", "Business coaching"]
        - "score", où vous devez évaluer, sur une échelle de 1 à 10, à quel point ce résumé est approprié pour former un modèle de langage de grande taille en coaching et développement personnel.
        - "raison", où vous devez expliquer le choix du score attribué dans "model_training", en maximum 3 mots-clefs.
        Répondez en JSON. Fournissez uniquement le 5ème résumé, le plus dense, ainsi que ses Entités Manquantes et la metadata correspondante. 
        Le JSON doit être un dictionnaire dont les clés sont "Entités_Manquantes", "Résumé", "Metadata".
        """
    
        chat = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
        prompt = ChatPromptTemplate.from_template(template=template)
        chains_dict = {}
        try:
            # Your existing code
            for i in tqdm(indices_to_process, desc='Generating Chains...'):
                txt = texts[i]
                key = f"chains_{i}"
                messages = prompt.format_messages(ARTICLE=txt, LEN=len(txt)/2, THEMES=themes)
                response = chat(messages)
                chains_dict[key] = response.content
                
            rows = []
            for key, value in chains_dict.items():
                data_dict = json.loads(value)
                metadata = data_dict.pop('Metadata')
                data_dict.update(metadata)
                data_dict['chain_index'] = key
                rows.append(data_dict)

            df2 = pd.DataFrame(rows)
            df2['texts'] = [texts[i] for i in indices_to_process]
            df2.to_csv(f'../TranscriptQAOutputs/COD/{name}_COD.csv')
            print("Done ✅")
            print(f'{name} successfully saved to ../TranscriptQAOutputs/COD/{name}_COD.csv ✅')
        except Exception as e:
            print(f"An error occurred: {e}")
            with open('chains_dict_error.txt', 'w') as f:
                f.write(json.dumps(chains_dict, indent=4))
    else:
        template = """Vous êtes un coach expert. Voici les transcriptions d'une masterclass de coaching.
        Vous résumerez chaque point et identifierez les concepts clés qui y sont énoncés. 
        Veillez à ce que les résumés ne dépassent pas {leng} caractères.

        Voici le texte:
        "{text}"

        Résumé:
        """
        chat = ChatOpenAI(temperature=0, model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
        prompt = ChatPromptTemplate.from_template(template=template)
        summaries = []
        for i in tqdm(indices_to_process, desc='Generating Summaries...'):
            txt = texts[i]
            messages = prompt.format_messages(text = txt,leng = len(i)//2)
            response = chat(messages)
            summaries.append(response.content)
        df2 = pd.DataFrame()
        df2['summaries'] = summaries
        df2['texts'] = [texts[i] for i in indices_to_process]
        df2.to_csv(f'../TranscriptQAOutputs/COD/{name}_COD.csv')
        print("Done ✅")
        print(f'{name} successfully saved to pdfprompts/TranscriptQAOutputs/COD/{name}.csv ✅')
    return df2