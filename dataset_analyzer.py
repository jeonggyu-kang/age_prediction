from collections import defaultdict

import pandas as pd 
import cv2
from tqdm import tqdm 

from config import get_hyperparameters 
from dataset import get_dataloader 
from evaluation import plot_dataset_dist
from utils import print_dataset_dist

def main():
    args = get_hyperparameters()
    
    # train dataset

    mode = 'train'
    train_loader = get_dataloader(
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = 1,
        num_workers = 1,
        pipeline = args[mode]['pipeline']
    )

    train_df = pd.read_parquet(args[mode]['ann_file'], engine='pyarrow').values

    # test dataset

    mode = 'test'
    test_loader = get_dataloader(
        data_dir = args[mode]['img_dir'],
        ann_path = args[mode]['ann_file'],
        mode = mode,
        batch_size = 1,
        num_workers = 1,
        pipeline = args[mode]['pipeline']
    )

    test_df = pd.read_parquet(args[mode]['ann_file'], engine='pyarrow').values

    print (train_df[0])
    print (test_df[0])

    train_sample_dict = defaultdict(int)

    pbar = tqdm(total = len(train_loader) + len(test_loader))
    for batch in train_loader:
        x, y = batch
        class_idx = y.item()
        train_sample_dict[class_idx] += 1

        pbar.update()

    test_sample_dict = dict()
    for k in train_sample_dict:
        test_sample_dict[k] = 0

    for batch in test_loader:
        x, y = batch
        class_idx = y.item()
        test_sample_dict[class_idx] += 1

        pbar.update()

    print_dataset_dist(train_sample_dict, 'train')
    print_dataset_dist(test_sample_dict, 'test')

    trainset_dist_img = plot_dataset_dist(train_sample_dict)
    testset_dist_img = plot_dataset_dist(test_sample_dict)

    cv2.imwrite('trainset_distribution.png', trainset_dist_img)
    cv2.imwrite('testset_distribution.png', testset_dist_img)


if __name__ == '__main__':
    main()

