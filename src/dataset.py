import re
import string
from enum import Enum
from pathlib import Path

import pandas as pd
import stop_words
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


class DatasetColumns(Enum):
    CLASS = 'Class Index'
    TITLE = 'Title'
    DESCRIPTION = 'Description'


def load_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(path)
    dataset = dataset.drop(DatasetColumns.TITLE.value, axis='columns')
    dataset[DatasetColumns.DESCRIPTION.value] = [line.split('\n')
                                                 for line in dataset[DatasetColumns.DESCRIPTION.value]]
    return dataset


def remove_stop_words(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset[DatasetColumns.DESCRIPTION.value] = [' '.join(
            [word if word not in stop_words.get_stop_words('en') else '' for word in text]
        ) for text in dataset[DatasetColumns.DESCRIPTION.value]]
    return dataset


def remove_symbols(dataset: pd.DataFrame) -> pd.DataFrame:
    exclude_symbols = u''.join(['№', '«', 'ђ', '°', '±', '‚', 'ћ', '‰', '…', '»', 'ѓ', 'µ', '·', 'ґ', 'њ', 'ї', 'џ', 'є', '‹',
                            '‡', '†', '¶', 'ќ', '€', '“', 'ў', '§', '„', '”', '\ufeff', '’', 'љ', '›', '•', '—', '‘',
                            '\x7f', '\xad', '¤', '\xa0', '\u200b', '–']) + string.punctuation + string.digits
    regex_symb = re.compile('[%s]' % re.escape(exclude_symbols))

    dataset[DatasetColumns.DESCRIPTION.value] = [regex_symb.sub(' ', ''.join(text))
                                                 for text in dataset[DatasetColumns.DESCRIPTION.value]]
    dataset[DatasetColumns.DESCRIPTION.value] = [re.sub(r' +', ' ', text) for text in dataset[DatasetColumns.DESCRIPTION.value]]
    return dataset


def make_vectorizer(dataset: pd.DataFrame) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(max_features=10000)
    vectorizer.fit(dataset[DatasetColumns.DESCRIPTION.value])
    return vectorizer


def tokenize(vectorizer: TfidfVectorizer, dataset: pd.DataFrame) -> pd.DataFrame:
    vectorized_data = vectorizer.transform(dataset[DatasetColumns.DESCRIPTION.value]).toarray()
    vectorized_data = [vector for vector in vectorized_data]
    dataset[DatasetColumns.DESCRIPTION.value] = vectorized_data
    return dataset


def split(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    (samples_train, samples_test,
     targets_train, targets_test) = train_test_split(dataset[DatasetColumns.DESCRIPTION.value],
                                                     dataset[DatasetColumns.CLASS.value],
                                                     test_size=0.15, random_state=42)
    train = pd.concat([targets_train, samples_train], axis='columns')
    test = pd.concat([targets_test, samples_test], axis='columns')
    return train, test


def rename_columns(dataset: pd.DataFrame) -> pd.DataFrame:
    if DatasetColumns.CLASS.value in dataset.columns:
        dataset['label'] = dataset.pop(DatasetColumns.CLASS.value)
    dataset['input_ids'] = dataset.pop(DatasetColumns.DESCRIPTION.value)
    return dataset


def collate_fn(input_batch: list[dict]) -> dict[str, torch.Tensor]:
    new_batch = {}

    sequences = torch.LongTensor([x['input_ids'] for x in input_batch])
    new_batch['input_ids'] = sequences

    if 'label' in input_batch[0]:
        labels = torch.LongTensor([x['label'] - 1 for x in input_batch])
        new_batch['label'] = labels

    return new_batch
