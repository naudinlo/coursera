import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from utils import *

import numpy as np


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(
            paths["WORD_EMBEDDINGS"]
        )
        self.thread_embeddings_folder = paths["THREAD_EMBEDDINGS_FOLDER"]

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        question_vec = question_to_vec(
            question, self.word_embeddings, self.embeddings_dim
        ).reshape(1, -1)
        best_thread = pairwise_distances_argmin(question_vec, thread_embeddings)

        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths["INTENT_RECOGNIZER"])
        self.tfidf_vectorizer = unpickle_file(paths["TFIDF_VECTORIZER"])

        self.ANSWER_TEMPLATE = "I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s"

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths["TAG_CLASSIFIER"])
        self.thread_ranker = ThreadRanker(paths)

        # Chitchat part

        self.create_chitchat_bot()

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param

        self.chitchat_bot = ChatBot(
            "Stack Over Bot", trainer="chatterbot.trainers.ChatterBotCorpusTrainer"
        )
        self.chitchat_bot.train("chatterbot.corpus.english")

    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.

        prepared_question = text_prepare(question)  #### YOUR CODE HERE ####
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)  #### YOUR CODE HERE ####

        # Chit-chat part:
        if intent == "dialogue":
            response = "Do you want to here a joke?"  #### YOUR CODE HERE ####
            response = self.chitchat_bot.get_response(question)
            return response

        # Goal-oriented part:
        else:
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)[0]  #### YOUR CODE HERE ####
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)[0]

            return self.ANSWER_TEMPLATE % (tag, thread_id)
