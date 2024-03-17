import re
from tqdm import tqdm
def anonymize(texts):
    """
    This function takes a list of texts and returns them anonymized (names replaced with 'PERSON').
    It replaces any capitalized names found in unique_names.txt for each text in the list.

    Args:
        texts: a list of strings

    Returns:
        anonymized_texts: a list of strings with capitalized names replaced by 'PERSON'
    """
    with open('../Resources/unique_names.txt', 'r', encoding='utf-8') as file:
        names_set = {line.strip().title() for line in file}
    anonymized_texts = []
    for text in tqdm(texts, desc='Anonymizing Texts...'):
        words_in_text = {word for word in re.findall(r'\b[A-Z][a-z]*\b', text)}
        names_to_replace = words_in_text.intersection(names_set)
        for name in names_to_replace:
            text = re.sub(r'\b' + re.escape(name) + r'\b', 'PERSON', text)
        anonymized_texts.append(text)
    return anonymized_texts