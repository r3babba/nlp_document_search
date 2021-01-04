# -*- coding: utf-8 -*-
"""
Document Search Engine using NLP models
"""

__author__ = "Rahul Kalubowila"
__version__ = "0.2.0"

from sentence_transformers import SentenceTransformer, util
import pandas as pd
import pickle
import torch
import os.path

def load_data(path):
    #Returns the sentence list

    #Parameters:
    #    path (str):The path of the CSV file
    
    #Returns:
    #    df(pd.DataFrame):The list with sentences
    
    # load corpus
    df = pd.read_csv(path, encoding='latin-1')
    df = df.rename(columns={df.columns.values[0]: "Title"})
   
    #Our sentences we like to encode
    sentences = df.Title.tolist()
    return sentences
    


def load_embeddings(embedding_to_open):
    #Returns sentences & embeddings from disc

    #Parameters:
    #    embedding_to_open (str):The path of the embedding .pkl file
    
    #Returns:
    #    stored_embeddings(Dictionary):The Dictionary with sentences and embeddings
    
    with open(embedding_to_open, "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_embeddings = stored_data['embeddings']
        return stored_embeddings



def search(query_list, sentences, stored_embeddings, model_name, k):
    #Returns closest k sentences of the corpus for each query sentence based on cosine similarity

    #Parameters:
    #    query_list (list):The list of queries to search
    #    sentences (list):The list of sentences in the corpus
    #    stored_embeddings (Dict):The Dictionary with sentences and embeddings
    #    model_name (str):The name of the chosen model
    #    k (int):The number of sentences retr
    
    #Returns:
    #    prints closest k sentences of the corpus for each query sentence based on cosine similarity

    top_k = k
    
    #model used 
    model = SentenceTransformer(model_name)
    for query in query_list:
        query_embedding = model.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, stored_embeddings)[0]
        cos_scores = cos_scores.cpu()
    
        #We use torch.topk to find the highest 5 scores
        top_results = torch.topk(cos_scores, k=top_k)
    
        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
    
        for score, idx in zip(top_results[0], top_results[1]):
            print(sentences[idx], "(Score: %.4f)" % (score))

def train_embeddings(sentences, model_name):
    #Trains a model with corpus and saves embeddings to disc

    #Parameters:
    #    sentences (list):The list of sentences in the corpus
    #    model_name (str):The name of the chosen model
    
    #Returns:
    #    saves embeddings to disc
   
    model = SentenceTransformer(model_name)
    
    #Sentences are encoded by calling model.encode()
    embeddings = model.encode(sentences, convert_to_tensor=True)
    
    #Path to save model
    embeddings_folder = os.path.join("embeddings")
    embedding_save_path = os.path.join(embeddings_folder, model_name+"_embeddings.pkl")
    
    #Store sentences & embeddings on disc
    with open(embedding_save_path, "wb") as fOut:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


def search_pretrained(query_list):
    #Main Search Function

    #Parameters:
    #    query_list (list):The list of queries to search
    
    #Returns:
    #    Top K Search Results
    corpus_path = 'Data\data - Copy.csv'
    sentences = load_data(corpus_path)
    model_name_list = ['stsb-roberta-large', 'msmarco-distilroberta-base-v2', 'paraphrase-distilroberta-base-v1']
    print('Choose an NLP model:')
    print('1.'+model_name_list[0]+'(Semantic Textual Similarity)')
    print('2.'+model_name_list[1]+'(Information Retrieval)')
    print('3.'+model_name_list[2]+'(Paraphrase Identification)')
    n = int(input("Enter Choice: "))
    model_name = model_name_list[n-1]
    embeddings_folder = os.path.join("embeddings")
    embedding_to_open = os.path.join(embeddings_folder, model_name+"_embeddings.pkl")
    stored_embeddings = load_embeddings(embedding_to_open)
    k = int(input("Enter Number of Results: "))
    search(query_list, sentences, stored_embeddings, model_name, k)


# Running main function
search_pretrained(['profits','construction','losses incurred'])