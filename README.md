# pdfprompts

16 output files were excluded because PDF pages couldn't be hashed after multiple attempts.

Params are the following:

    chat = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo")

    template = """
    Tu es un coach d'exception. Génère cinq questions différentes en français basées sur ce document. Pour chacune des cinq questions, génère une réponse très détaillée

    {page}

    {format_instructions}
    """
