import json
import os
import glob
import tqdm
import jsonlines
import pandas as pd
import re
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything


def parse_json_column(genre_data):
    """
    Read genre information as a json string and convert it to a dict
    :param genre_data: genre data to be converted
    :return: dict of genre names
    """
    try:
        return json.loads(genre_data)
    except Exception as e:
        return None  # when genre information is missing


def load_booksummaries_data(book_path):
    """
    Load the Book Summary data and split it into train/dev/test sets
    :param book_path: path to the booksummaries.txt file
    :return: train, dev, test as pandas data frames
    """
    book_df = pd.read_csv(book_path, sep='\t', names=["Wikipedia article ID",
                                                      "Freebase ID",
                                                      "Book title",
                                                      "Author",
                                                      "Publication date",
                                                      "genres",
                                                      "summary"],
                           converters={'genres': parse_json_column})
    book_df = book_df.dropna(subset=['genres', 'summary'])  # remove rows missing any genres or summaries
    book_df['word_count'] = book_df['summary'].str.split().str.len()
    book_df = book_df[book_df['word_count'] >= 10]
    train = book_df.sample(frac=0.8, random_state=22)
    rest = book_df.drop(train.index)
    dev = rest.sample(frac=0.5, random_state=22)
    test = rest.drop(dev.index)
    return train, dev, test


def prepare_book_summaries(pairs, book_path='C:/Users/Monika/Downloads/BookSummaries/BookSummaries/data/booksummaries.txt'):
    """
    Load the Book Summary data and prepare the datasets
    :param pairs: whether to combine pairs of documents or not
    :param book_path: path to the booksummaries.txt file
    :return: dicts of lists of documents and labels and number of labels
    """
    if not os.path.exists(book_path):
        raise Exception("Data not found: {}".format(book_path))
    
    text_set = {'train': [], 'dev': [], 'test': []}
    label_set = {'train': [], 'dev': [], 'test': []}
    
    train, dev, test = load_booksummaries_data(book_path)
    
    if not pairs:
        text_set['train'] = train['summary'].tolist()
        text_set['dev'] = dev['summary'].tolist()
        text_set['test'] = test['summary'].tolist()
        
        train_genres = train['genres'].tolist()
        label_set['train'] = [list(genre.values()) for genre in train_genres]
        
        dev_genres = dev['genres'].tolist()
        label_set['dev'] = [list(genre.values()) for genre in dev_genres]
        
        test_genres = test['genres'].tolist()
        label_set['test'] = [list(genre.values()) for genre in test_genres]
    else:
        train_temp = train['summary'].tolist()
        dev_temp = dev['summary'].tolist()
        test_temp = test['summary'].tolist()
        
        train_genres = train['genres'].tolist()
        train_genres_temp = [list(genre.values()) for genre in train_genres]
        
        dev_genres = dev['genres'].tolist()
        dev_genres_temp = [list(genre.values()) for genre in dev_genres]
        
        test_genres = test['genres'].tolist()
        test_genres_temp = [list(genre.values()) for genre in test_genres]
        
        for i in range(0, len(train_temp) - 1, 2):
            text_set['train'].append(train_temp[i] + train_temp[i+1])
            label_set['train'].append(list(set(train_genres_temp[i] + train_genres_temp[i+1])))
        
        for i in range(0, len(dev_temp) - 1, 2):
            text_set['dev'].append(dev_temp[i] + dev_temp[i+1])
            label_set['dev'].append(list(set(dev_genres_temp[i] + dev_genres_temp[i+1])))
        
        for i in range(0, len(test_temp) - 1, 2):
            text_set['test'].append(test_temp[i] + test_temp[i+1])
            label_set['test'].append(list(set(test_genres_temp[i] + test_genres_temp[i+1])))
    
    vectorized_labels, num_labels = vectorize_labels(label_set)
    return text_set, vectorized_labels, num_labels


def vectorize_labels(all_labels):
    """
    Combine labels across all data and reformat the labels e.g. [[1, 2], ..., [123, 343, 4] ] --> [[0, 1, 1, ... 0], ...]
    Only used for multi-label classification
    :param all_labels: dict with labels with keys 'train', 'dev', 'test'
    :return: dict of vectorized labels per split and total number of labels
    """
    all_set = []
    for split in all_labels:
        for labels in all_labels[split]:
            all_set.extend(labels)
    
    all_set = list(set(all_set))
    
    mlb = MultiLabelBinarizer()
    mlb.fit([all_set])
    num_labels = len(mlb.classes_)
    print(f'Total number of labels: {num_labels}')
    
    result = {}
    for split in all_labels:
        result[split] = mlb.transform(all_labels[split])
    
    return result, num_labels


if __name__ == "__main__":
    seed_everything(3456)
    
    book_text_set, book_label_set, book_num_labels = prepare_book_summaries(False)
    assert book_num_labels == 227
    assert len(book_text_set['train']) == len(book_label_set['train']) == 10230
    assert len(book_text_set['dev']) == len(book_label_set['dev']) == 1279
    assert len(book_text_set['test']) == len(book_label_set['test']) == 1279
    
    pair_text_set, pair_label_set, pair_num_labels = prepare_book_summaries(True)
    assert pair_num_labels == 227
    assert len(pair_text_set['train']) == len(pair_label_set['train']) == 5115
    assert len(pair_text_set['dev']) == len(pair_label_set['dev']) == 639
    assert len(pair_text_set['test']) == len(pair_label_set['test']) == 639

