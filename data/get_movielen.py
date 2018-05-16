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

_raw_folder = 'raw'
_processed_folder = 'processed'
_file_list = ['train-ratings.csv', 'test-ratings.csv', 'test-negative.csv']
_TRAIN_RATINGS_FILENAME = 'train-ratings.csv'
_TEST_RATINGS_FILENAME = 'test-ratings.csv'
_TEST_NEG_FILENAME = 'test-negative.csv'

RatingData = namedtuple('RatingData',
                        ['items', 'users', 'ratings', 'min_date', 'max_date'])


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-r', '--root', type=str, default=os.path.join(os.getcwd(), 'data'))
    parser.add_argument('-n', '--num_neg', type=int, default=100)
    parser.add_argument('-s', '--seed', type=int, default=0)
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


def download(root):
    """Download the movielen data if it doesn't exist in processed_folder already."""
    from six.moves import urllib
    import gzip

    url = 'http://files.grouplens.org/datasets/movielens/ml-20m.zip' if len(sys.argv) > 3 and sys.argv[2] == '20m' \
        else 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'

    if _check_exists(root, _processed_folder, _file_list):
        return

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
    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    file_path = os.path.join(root, _raw_folder, filename)
    with open(file_path, 'wb') as f:
        f.write(data.read())
    with open(file_path.replace('.gz', ''), 'wb') as out_f, \
            gzip.GzipFile(file_path) as zip_f:
        out_f.write(zip_f.read())
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
    ratings = pd.read_csv(filename, sep='::', names=names, engine='python')
    return process_movielens(ratings, sort=sort)


def load_ml_10m(filename, sort=True):
    names = ['user_id', 'item_id', 'rating', 'timestamp']
    ratings = pd.read_csv(filename, sep='::', names=names, engine='python')
    return process_movielens(ratings, sort=sort)


def load_ml_20m(filename, sort=True):
    ratings = pd.read_csv(filename)
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


if __name__ == "__main__":
    args = parse_args()
    np.random.seed(args.seed)
    fn = download(args.root)
    fp = os.path.join(args.root, _raw_folder, fn)

    if not _check_exists(args.root, _raw_folder, fn):
        raise RuntimeError('Dataset not found. It may caused by download error.' +
                           ' You shall checkout you web status to download it')

    USER_COLUMN = 'user_id'
    ITEM_COLUMN = 'item_id'

    df = implicit_load(fp, sort=False)
    grouped = df.groupby(USER_COLUMN)
    df = grouped.filter(lambda x: len(x) >= 20)

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
    for row in df.itertuples():
        user_to_items[getattr(row, USER_COLUMN)].append(getattr(row, ITEM_COLUMN))  # noqa: E501

    test_ratings = []
    test_negs = []
    all_items = set(range(len(original_items)))
    for user in range(len(original_users)):
        test_item = user_to_items[user].pop()

        all_ratings.remove((user, test_item))
        all_negs = all_items - set(user_to_items[user])
        all_negs = sorted(list(all_negs))  # determinism

        test_ratings.append((user, test_item))
        test_negs.append(list(np.random.choice(all_negs, args.num_neg)))

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

