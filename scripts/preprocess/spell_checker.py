from spellchecker import SpellChecker
import spacy
nlp = spacy.load("en_core_web_sm")

spell = SpellChecker()

# find those words that may be misspelled

misspelled = spell.unknown(["LtTNGS","are","qLiick","y","RELIEVIII"])

sentences = [["LtTNGS","are","qLiick","y","RELIEVIII"], ["meeting","of","deleg4tes","and","represntativ3s","of","laour","assembled"]]


for sentence in sentences:
    newsentence = []
    for word in sentence:
        word = spell.correction(word)
        newsentence.append(word)
    print(newsentence)