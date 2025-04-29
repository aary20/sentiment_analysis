import PyPDF2
import spacy

nlp = spacy.load("en_core_web_sm")  # load small English NLP model

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_skills(text):
    doc = nlp(text)
    skills = set()

    for ent in doc.ents:
        if ent.label_ in ["SKILL", "ORG", "PERSON"]:  # Customize depending on what you detect
            skills.add(ent.text)

    return list(skills)
