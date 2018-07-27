#!/usr/bin/env python
# -- coding: utf-8 --
"""
Created by H. L. Wang on 2018/5/15

"""
import os
import sys
import errno
from collections import namedtuple, defaultdict
from argparse import ArgumentParser
import pandas as pd
import numpy as np
import zipfile
from tqdm import tqdm

_raw_folder = 'raw'
_processed_folder = 'processed'
_dataset_list = {'ml-100k', 'ml-1m', 'ml-10m', 'ml-20m'}
_file_list = ['train-ratings.csv', 'test-ratings.csv', 'test-negative.csv']
_raw_file_list = ['movies.dat', 'ratings.dat', 'users.dat']
_TRAIN_RATINGS_FILENAME = 'train-ratings.csv'
_TEST_RATINGS_FILENAME = 'test-ratings.csv'
_TEST_NEG_FILENAME = 'test-negative.csv'
MIN_RATINGS = 20

RatingData = namedtuple('RatingData',
                        ['items', 'users', 'ratings', 'min_date', 'max_date'])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default=os.path.join(os.getcwd(), 'data'),
                        help='Output directory for train and test CSV files')
    parser.add_argument('-n', '--num_neg', type=int, default=999,
                        help='Number of negative samples for each positive test example')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed to reproduce same negative samples')
    parser.add_argument('-d', '--dataset', type=str, default='ml-1m',
                        help='The Dataset used to train, currently support {ml-100k, ml-1m, ml-10m, ml-20m}')
    return parser.parse_args()


def describe_ratings(ratings):
    info = RatingData(items=len(ratings['item_id'].unique()),
                      users=len(ratings['user_id'].unique()),
                      ratings=len(ratings),
                      min_date=ratings['timestamp'].min(),
                      max_date=ratings['timestamp'].max())
    print("{ratings} ratings on {items} items from {users} users"
          " from {min_date} to {max_date}"
            .format(**(info._asdict())))
    return info


def _check_exists(root, processed_folder, file_list):
    return all([os.path.exists(os.path.join(root, processed_folder, file)) for file in file_list])


def download(root, args):
    """Download the movielen data if it doesn't exist in processed_folder already."""
    import requests

    assert args.dataset in _dataset_list

    url = 'http://files.grouplens.org/datasets/movielens/' + args.dataset + '.zip'

    if _check_exists(root, _processed_folder, _file_list):
        return url.rpartition('/')[2]

    # download files
    try:
        os.makedirs(os.path.join(root, _raw_folder))
        os.makedirs(os.path.join(root, _processed_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    print('Downloading ' + url)
    r = requests.get(url, stream=True)
    file_size = int(r.headers["Content-Length"])
    chunk_size = 1024
    bars = int(file_size / chunk_size)

    filename = url.rpartition('/')[2]
    file_path = os.path.join(root, _raw_folder, filename)

    if os.path.exists(file_path):
        exist_size = os.path.getsize(file_path)
        if exist_size == file_size:
            return filename

    with open(file_path, "wb") as f:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=bars, unit="KBytes",
                          desc=filename, leave=True):
            f.write(chunk)
    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(os.path.join(root, _raw_folder))
    os.unlink(file_path)
    return filename


def process_movielens(ratings, sort=True):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    if sort:
        ratings.sort_values(by='timestamp', inplace=True)
    describe_ratings(ratings)
    return ratings


def load_ml_100k(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='\t', names=names)
    return process_movielens(ratings, sort=sort)


def load_ml_1m(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename+".dat", sep='::', names=names, engine='python')
    return process_movielens(ratings, sort=sort)


def load_ml_10m(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='::', names=names, engine='python')
    return process_movielens(ratings, sort=sort)


def load_ml_20m(filename, sort=True):
    ratings = pd.read_csv(filename+".csv")
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')
    names = {'userId': 'user_id', 'movieId': 'item_id'}
    ratings.rename(columns=names, inplace=True)
    return process_movielens(ratings, sort=sort)


DATASETS = [k.replace('load_', '') for k in locals().keys() if "load_" in k]


def get_dataset_name(filename):
    for dataset in DATASETS:
        if dataset in filename.replace('-', '_').lower():
            return dataset
    raise NotImplementedError


def implicit_load(filename, sort=True):
    func = globals()["load_" + get_dataset_name(filename)]
    return func(filename, sort=sort)


def convert(fn):
    fp = os.path.join(args.root, _raw_folder, fn, 'ratings')
    if not _check_exists(args.root, os.path.join(_raw_folder, fn), _raw_file_list):
        raise RuntimeError('Dataset not found. It may caused by download error.' +
                           ' You shall checkout you web status to download it')

    USER_COLUMN = 'user_id'
    ITEM_COLUMN = 'item_id'
    RATE_COLUMN = 'rating'

    df = implicit_load(fp, sort=False)
    grouped = df.groupby(USER_COLUMN)
    df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

    original_users = df[USER_COLUMN].unique()
    original_items = df[ITEM_COLUMN].unique()

    user_map = {user: index for index, user in enumerate(original_users)}
    item_map = {item: index for index, item in enumerate(original_items)}

    df[USER_COLUMN] = df[USER_COLUMN].apply(lambda user: user_map[user])
    df[ITEM_COLUMN] = df[ITEM_COLUMN].apply(lambda item: item_map[item])

    assert df[USER_COLUMN].max() == len(original_users) - 1
    assert df[ITEM_COLUMN].max() == len(original_items) - 1

    # Need to sort before popping to get last item
    df.sort_values(by='timestamp', inplace=True)
    all_ratings = set(zip(df[USER_COLUMN], df[ITEM_COLUMN]))
    user_to_items = defaultdict(list)
    for row in tqdm(df.itertuples(), desc='Ratings', total=len(df)):
        user_to_items[getattr(row, USER_COLUMN)].append(getattr(row, ITEM_COLUMN))  # noqa: E501
    try:
        print(f"Generating {args.num_neg} negative samples for each user")
    except:
        print("Generating {} negative samples for each user".format(args.negatives))

    test_ratings = []
    test_negs = []
    all_items = set(range(len(original_items)))
    for user in tqdm(range(len(original_users)), desc='Users', total=len(original_users)):
        test_item = user_to_items[user].pop()

        all_ratings.remove((user, test_item))
        all_negs = all_items - set(user_to_items[user])
        all_negs = sorted(list(all_negs))  # determinism

        test_ratings.append((user, test_item))
        test_negs.append(list(np.random.choice(all_negs, args.num_neg)))

    return all_ratings, test_ratings, test_negs


def save(all_ratings, test_ratings, test_negs):
    # serialize
    df_train_ratings = pd.DataFrame(list(all_ratings))
    df_train_ratings['fake_rating'] = 1
    df_train_ratings.to_csv(os.path.join(args.root, _processed_folder, _TRAIN_RATINGS_FILENAME),
                            index=False, header=False, sep='\t')

    df_test_ratings = pd.DataFrame(test_ratings)
    df_test_ratings['fake_rating'] = 1
    df_test_ratings.to_csv(os.path.join(args.root, _processed_folder, _TEST_RATINGS_FILENAME),
                           index=False, header=False, sep='\t')

    df_test_negs = pd.DataFrame(test_negs)
    df_test_negs.to_csv(os.path.join(args.root, _processed_folder, _TEST_NEG_FILENAME),
                        index=False, header=False, sep='\t')


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    fn = download(args.root, args).replace(".zip", "")
    all_ratings, test_ratings, test_negs = convert(fn)
    save(all_ratings, test_ratings, test_negs)
