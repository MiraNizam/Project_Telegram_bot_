import random
import re
import nltk
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


def filter_func(text):
    """ The function receives text, converts it to single case and removes everything except letters and spaces"""
    expression = r"[^\w\s]"
    return re.sub(expression, "", text.lower())


def text_match(user_text, example):
    """ Processes by Filter user_text, example. Checks for an empty string """
    user_text = filter_func(user_text)
    example = filter_func(example)

    if len(user_text) == 0 or len(example) == 0:
        return False

    distance = nltk.edit_distance(user_text, example) / len(example)
    return distance < 0.2


with open("big_bot_config.json", "r") as config_file:
    BIG_INTENTS = json.load(config_file)

INTENTS = BIG_INTENTS["intents"]
"""
Classification of texts, the model determines the intent according to the user's text
phrases_x - collects "examples", "responses"
intents_y - collects intents
"""
phrases_x = []
intents_y = []

for name, intent in INTENTS.items():
    for phrase in intent["examples"]:
        phrases_x.append(phrase)
        intents_y.append(name)

    for phrase in intent["responses"]:
        phrases_x.append(phrase)
        intents_y.append(name)

"""We start using Sklearn module to vectorize the texts."""
vectorizer_phrases = CountVectorizer()
# vectorizer_phrases = TfidfVectorizer()
vectorizer_phrases.fit(phrases_x)

"""Create model, Vectorize it, """
mlp_model = MLPClassifier()
vector_phrases = vectorizer_phrases.transform(phrases_x)
mlp_model.fit(vector_phrases, intents_y)

rf_model = RandomForestClassifier()
rf_model.fit(vector_phrases, intents_y)
rf_model.score(vector_phrases, intents_y)

MODEL = rf_model  # mlp_model


def get_intent_ml(text):
    vec_text = vectorizer_phrases.transform([text])
    intent = MODEL.predict(vec_text)[0]
    return intent


failure_phrases = BIG_INTENTS['failure_phrases']

INTENTS = BIG_INTENTS['intents']

def get_intent(text):
    """It defines the intents by text"""

    for intent_name in INTENTS.keys():
        examples = INTENTS[intent_name]["examples"]
        for example in examples:
            if text_match(text, example):
                return intent_name


def get_response(intent):
    return random.choice(INTENTS[intent]["responses"])


def bot(text):
    text = filter_func(text)
    intent = get_intent(text)  # Looking for intent
    if not intent:  # If the intent was not found
        intent = get_intent_ml(text)  # We involve ML model
    if intent:  # If we find something, print result
        return get_response(intent)
    else:
        return random.choice(failure_phrases)  # else, we use random plug

# text = ""
# while text != "Выход":
#     text = input()
#     print(bot(text))