# the following installations are required
# python3 -m textblob.download_corpora
# python3 -m spacy download en_core_web_sm

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe('spacytextblob')

import numpy as np


def build_featureset(data, normalize = True):

    feature_set = []
    subset_size = 100000

    for message in data['message'][:subset_size]:
        number_of_chars         = len(message)
        number_of_words         = len(nlp(message)) # word length? should normally be contained by number_of_chars / number_of_words
        number_uppercase_chars  = sum(1 for c in message if c.isupper())
        number_lowercase_char   = sum(1 for c in message if c.islower())
        point_count             = message.count(".")
        comma_count             = message.count(",")
        colon_count             = message.count(":")
        dash_count              = message.count("-")
        line_count              = message.count("_")
        hashtag_count           = message.count("#")
        open_bracket_1_count    = message.count('(')
        close_bracket_1_count   = message.count(')')
        open_bracket_2_count    = message.count('[')
        close_bracket_2_count   = message.count(']')
        open_bracket_3_count    = message.count('{')
        close_bracket_3_count   = message.count('}')
        smaller_sign_count      = message.count('<')
        bigger_sign_count       = message.count('>')
        slash_count             = message.count('/')
        back_slash_count        = message.count('\\')
        polarity                = nlp(message)._.blob.polarity
        subjectivity            = nlp(message)._.blob.subjectivity
        first_tag               = nlp(message)[0].pos # do we need to encode this ?
        second_tag  = 0
        third_tag   = 0
        if len(nlp(message)) > 1:
            second_tag          = nlp(message)[1].pos # do we need to encode this ?
        if len(nlp(message)) > 2:
            third_tag           = nlp(message)[2].pos # do we need to encode this ?
        feature_set.append([
                number_of_chars, number_of_words,
                number_uppercase_chars, number_lowercase_char,
                point_count, comma_count, colon_count, dash_count, line_count, hashtag_count, 
                open_bracket_1_count, close_bracket_1_count, 
                open_bracket_2_count, close_bracket_2_count, 
                open_bracket_3_count, close_bracket_3_count,
                smaller_sign_count, bigger_sign_count,
                slash_count, back_slash_count,
                polarity, subjectivity,
                first_tag, second_tag, third_tag
                ])
        
    if normalize:
        feature_set = feature_set / np.linalg.norm(feature_set)

    return feature_set