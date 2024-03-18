# PDF Document Parser and Q&A Generator

This repository contains a tool designed to process PDF documents and convert them into an easily navigable question and answer format. The tool automates the extraction of text from PDFs, breaking it down into coherent paragraphs, generating relevant questions, and providing corresponding answers. The process ensures high data quality through a chain of density methodology that refines the content. Additionally, the tool incorporates an anonymization process to protect sensitive information.

## Features

- **PDF Parsing**: Converts PDF documents into text paragraphs.
- **Paragraph Optimization**: Utilizes a chain of density algorithm with LangChain to refine paragraphs for clarity and quality.
- **Question Generation**: Automatically generates questions based on the content of the paragraphs.
- **Answer Generation**: Provides answers to the generated questions.
- **Data Anonymization**: Ensures that personal or sensitive data within the documents is anonymized for privacy.

## Installation

To get started with this tool, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
pip install -r requirements.txt
```

## Usage

Run the main script to process your PDFs:

```bash
python main.py <path_to_your_pdf>
```

The script will process the document and output the questions and answers, along with the optimized paragraphs, in a structured format.

## Modules

- `main.py`: The entry point for the tool, coordinating the PDF processing and Q&A generation.
- `anonymizer.py`: Module responsible for detecting and anonymizing sensitive data.
- `paragrapher.py`: Handles the breakdown of text into optimized paragraphs.
- `questionanswerer.py`: Generates questions and answers based on the paragraphs.
- `summarizer.py`: Summarizes the content to support the Q&A process.
