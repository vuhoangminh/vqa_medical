"""
Preprocess a train/test pair of interim json data files.
Caption: Use NLTK or split function to get tokens. 
"""
import os.path
import argparse
import numpy as np
import scipy.io
import pdb
import json
import csv
import re
import math
import pickle
import datasets.utils.process_qa_utils as process_utils
import datasets.utils.paths_utils as path_utils


def vqa_processed(params):
    
    #####################################################
    ## Read input files
    #####################################################

    path_train = os.path.join(params['dir'], 'interim', params['trainsplit']+'_questions_annotations.json')
    if params['trainsplit'] == 'train':
        path_val = os.path.join(params['dir'], 'interim', 'val_questions_annotations.json')
    path_test    = os.path.join(params['dir'], 'interim', 'test_questions.json')
    path_testdev = os.path.join(params['dir'], 'interim', 'testdev_questions.json')
 
    # An example is a tuple (question, image, answer)
    # /!\ test and test-dev have no answer
    trainset = json.load(open(path_train, 'r'))
    if params['trainsplit'] == 'train':
        valset = json.load(open(path_val, 'r'))
    testset    = json.load(open(path_test, 'r'))
    testdevset = json.load(open(path_testdev, 'r'))

    #####################################################
    ## Preprocess examples (questions and answers)
    #####################################################

    top_answers = process_utils.get_top_answers(trainset, params['nans'])
    aid_to_ans = [a for i,a in enumerate(top_answers)]
    ans_to_aid = {a:i for i,a in enumerate(top_answers)}
    # Remove examples if answer is not in top answers
    trainset = process_utils.remove_examples(trainset, ans_to_aid)

    # Add 'question_words' to the initial tuple
    trainset = process_utils.preprocess_questions(trainset, params['nlp'])
    if params['trainsplit'] == 'train':
        valset = process_utils.preprocess_questions(valset, params['nlp'])
    testset    = process_utils.preprocess_questions(testset, params['nlp'])
    testdevset = process_utils.preprocess_questions(testdevset, params['nlp'])

    # Also process top_words which contains a UNK char
    trainset, top_words = process_utils.remove_long_tail_train(trainset, params['minwcount'])
    wid_to_word = {i+1:w for i,w in enumerate(top_words)}
    word_to_wid = {w:i+1 for i,w in enumerate(top_words)}

    if params['trainsplit'] == 'train':
        valset = process_utils.remove_long_tail_test(valset, word_to_wid)
    testset    = process_utils.remove_long_tail_test(testset, word_to_wid)
    testdevset = process_utils.remove_long_tail_test(testdevset, word_to_wid)

    trainset = process_utils.encode_question(trainset, word_to_wid, params['maxlength'], params['pad'])
    if params['trainsplit'] == 'train':
        valset = process_utils.encode_question(valset, word_to_wid, params['maxlength'], params['pad'])
    testset    = process_utils.encode_question(testset, word_to_wid, params['maxlength'], params['pad'])
    testdevset = process_utils.encode_question(testdevset, word_to_wid, params['maxlength'], params['pad'])

    trainset = process_utils.encode_answer(trainset, ans_to_aid)
    trainset = process_utils.encode_answers_occurence(trainset, ans_to_aid)
    if params['trainsplit'] == 'train':
        valset = process_utils.encode_answer(valset, ans_to_aid)
        valset = process_utils.encode_answers_occurence(valset, ans_to_aid)

    #####################################################
    ## Write output files
    #####################################################

    # Paths to output files
    # Ex: data/vqa/processed/nans,3000_maxlength,15_..._trainsplit,train_testsplit,val/id_to_word.json
    subdirname = 'nans,'+str(params['nans'])
    for param in ['maxlength', 'minwcount', 'nlp', 'pad', 'trainsplit']:
        subdirname += '_' + param + ',' + str(params[param])
    os.system('mkdir -p ' + os.path.join(params['dir'], 'processed', subdirname))

    dir_save = os.path.join(params['dir'], 'processed', subdirname)
    if not os.path.exists(dir_save):
        print('>> make dir', dir_save)
        path_utils.make_dir(dir_save)

    path_wid_to_word = os.path.join(params['dir'], 'processed', subdirname, 'wid_to_word.pickle')
    path_word_to_wid = os.path.join(params['dir'], 'processed', subdirname, 'word_to_wid.pickle')
    path_aid_to_ans  = os.path.join(params['dir'], 'processed', subdirname, 'aid_to_ans.pickle')
    path_ans_to_aid  = os.path.join(params['dir'], 'processed', subdirname, 'ans_to_aid.pickle')
    if params['trainsplit'] == 'train':
        path_trainset = os.path.join(params['dir'], 'processed', subdirname, 'trainset.pickle')
        path_valset   = os.path.join(params['dir'], 'processed', subdirname, 'valset.pickle')
    elif params['trainsplit'] == 'trainval':
        path_trainset = os.path.join(params['dir'], 'processed', subdirname, 'trainvalset.pickle')
    path_testset     = os.path.join(params['dir'], 'processed', subdirname, 'testset.pickle')
    path_testdevset  = os.path.join(params['dir'], 'processed', subdirname, 'testdevset.pickle')

    print('Write wid_to_word to', path_wid_to_word)
    with open(path_wid_to_word, 'wb') as handle:
        pickle.dump(wid_to_word, handle)

    print('Write word_to_wid to', path_word_to_wid)
    with open(path_word_to_wid, 'wb') as handle:
        pickle.dump(word_to_wid, handle)

    print('Write aid_to_ans to', path_aid_to_ans)
    with open(path_aid_to_ans, 'wb') as handle:
        pickle.dump(aid_to_ans, handle)

    print('Write ans_to_aid to', path_ans_to_aid)
    with open(path_ans_to_aid, 'wb') as handle:
        pickle.dump(ans_to_aid, handle)

    print('Write trainset to', path_trainset)
    with open(path_trainset, 'wb') as handle:
        pickle.dump(trainset, handle)

    if params['trainsplit'] == 'train':
        print('Write valset to', path_valset)
        with open(path_valset, 'wb') as handle:
            pickle.dump(valset, handle)

    print('Write testset to', path_testset)
    with open(path_testset, 'wb') as handle:
        pickle.dump(testset, handle)

    print('Write testdevset to', path_testdevset)
    with open(path_testdevset, 'wb') as handle:
        pickle.dump(testdevset, handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirname',
        default='data/vqa_tools',
        type=str,
        help='Root directory containing raw, interim and processed directories'
    )
    parser.add_argument('--trainsplit',
        default='train',
        type=str,
        help='Options: train | trainval'
    )
    parser.add_argument('--nans',
        default=3,
        type=int,
        help='Number of top answers for the final classifications'
    )
    parser.add_argument('--maxlength',
        default=26,
        type=int,
        help='Max number of words in a caption. Captions longer get clipped'
    )
    parser.add_argument('--minwcount',
        default=0,
        type=int,
        help='Words that occur less than that are removed from vocab'
    )
    parser.add_argument('--nlp',
        default='mcb',
        type=str,
        help='Token method ; Options: nltk | mcb | naive'
    )
    parser.add_argument('--pad',
        default='left',
        type=str,
        help='Padding ; Options: right (finish by zeros) | left (begin by zeros)'
    )
    parser.add_argument('--dir',
        default='data/vqa_tools',
        type=str
    )
    args = parser.parse_args()
    opt_vqa = vars(args)
    vqa_processed(opt_vqa)


if __name__ == "__main__":
    main()
    