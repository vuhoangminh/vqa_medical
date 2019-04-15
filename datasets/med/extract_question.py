import vqa.models as models
import vqa.datasets as datasets
import vqa.lib.criterions as criterions
import vqa.lib.logger as logger
import vqa.lib.utils as utils
import vqa.lib.engine as engine
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
from pprint import pprint
import yaml
import pandas as pd
import argparse
from vqa.models import sen2vec
import os
import datasets.utils.paths_utils as path_utils
import datasets.utils.io_utils as io_utils
import datasets.utils.print_utils as print_utils
from vqa.models import sen2vec


parser = argparse.ArgumentParser(
    description='Train/Evaluate models',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
##################################################
# yaml options file contains all default choices #
# parser.add_argument('--path_opt', default='options/breast/default.yaml', type=str,
#                     help='path to a yaml options file')
parser.add_argument('--path_opt', default='options/med/minhmul_att_train_imagenet_h200_g8_relu_bert.yaml', type=str,
                    # parser.add_argument('--path_opt', default='options/med/minhmul_att_train_imagenet_h200_g8_relu.yaml', type=str,
                    help='path to a yaml options file')
################################################
# change cli options to modify default choices #
# logs options
parser.add_argument('--dir_logs', type=str, help='dir logs')
# data options
parser.add_argument('--vqa_trainsplit', type=str,
                    choices=['train', 'trainval'], default="train")
# model options
parser.add_argument('--arch', choices=models.model_names,
                    help='vqa model architecture: ' +
                    ' | '.join(models.model_names))
parser.add_argument('--st_type',
                    help='skipthoughts type')
parser.add_argument('--st_dropout', type=float)
parser.add_argument('--st_fixed_emb', default=None, type=utils.str2bool,
                    help='backprop on embedding')
# optim options
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='initial learning rate')
parser.add_argument('-b', '--batch_size', type=int,
                    help='mini-batch size')
parser.add_argument('--epochs', type=int,
                    help='number of total epochs to run')
# options not in yaml file
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint')
parser.add_argument('--save_model', default=True, type=utils.str2bool,
                    help='able or disable save model and optim state')
parser.add_argument('--save_all_from', type=int,
                    help='''delete the preceding checkpoint until an epoch,'''
                         ''' then keep all (useful to save disk space)')''')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation and test set')
parser.add_argument('-j', '--workers', default=0, type=int,
                    help='number of data loading workers')
parser.add_argument('--print_freq', '-p', default=10, type=int,
                    help='print frequency')
################################################
parser.add_argument('-ho', '--help_opt', dest='help_opt', action='store_true',
                    help='show selected options before running')


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"
EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features.pickle"


def main():

    df = pd.read_csv(PROCESSED_QA_PER_QUESTION_PATH)
    list_questions = list(df.question)
    list_questions = list(set(list_questions))

    #########################################################################################
    # Begin training on train/val or trainval/test
    #########################################################################################

    dict = {}
    batch = 16
    for i in range(0,len(list_questions),batch):
        fr, to = i, i+batch
        if to >= len(list_questions): 
            to = len(list_questions)
        print_utils.print_tqdm(to, len(list_questions), cutoff=2)
        questions = list_questions[fr:to]
        input_question = sen2vec.sen2vec(questions)
        for q, question in enumerate(questions):
            dict[question] = input_question[q].cpu().detach().numpy()
        # print(len(dict))
        del questions, input_question
        torch.cuda.empty_cache()

    io_utils.write_pickle(dict, EXTRACTED_QUES_FEATURES_PATH)


if __name__ == '__main__':
    main()
