import argparse
import os
import shutil
import yaml
import json
import click
from pprint import pprint

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd

import vqa.lib.utils as gen_utils
import datasets.utils.paths_utils as path_utils
import datasets.utils.io_utils as io_utils
import datasets.utils.metrics_utils as metrics_utils
import vqa.lib.engine as engine
import vqa.lib.utils as utils
import vqa.lib.logger as logger
import vqa.lib.criterions as criterions
import vqa.datasets as datasets
import vqa.models as models


SUB_SKIP_THOUGHTS = [
    "mutan_att_train_imagenet_relu",
    "mlb_att_train_imagenet_h200_g4_relu",
    "mlb_att_train_imagenet_h100_g8_relu",
    "globalbilinear_att_train_imagenet_h200_g4_relu",
    "globalbilinear_att_train_imagenet_h100_g8_relu",
]


SUB_BERT_3072 = [
    "mutan_att_train_imagenet_relu_bert_cased",
    "mutan_att_train_imagenet_relu_bert_uncased",
    "mlb_att_train_imagenet_h200_g4_relu_bert_uncased",
    "mlb_att_train_imagenet_h200_g4_relu_bert_cased",
    "mlb_att_train_imagenet_h100_g8_relu_bert_uncased",
    "mlb_att_train_imagenet_h100_g8_relu_bert_cased",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_uncased",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_cased",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_uncased",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_cased",
]


SUB_BERT_768 = [
    "mutan_att_train_imagenet_relu_bert_cased_768",
    "mutan_att_train_imagenet_relu_bert_uncased_768",
    "mlb_att_train_imagenet_h200_g4_relu_bert_uncased_768",
    "mlb_att_train_imagenet_h200_g4_relu_bert_cased_768",
    "mlb_att_train_imagenet_h100_g8_relu_bert_uncased_768",
    "mlb_att_train_imagenet_h100_g8_relu_bert_cased_768",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_uncased_768",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_cased_768",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_uncased_768",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_cased_768",
]


SUB_GLOBAL_BILINEAR = [
    "globalbilinear_att_train_imagenet_h200_g4_relu",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_uncased",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_cased",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_uncased_768",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_cased_768",
    "globalbilinear_att_train_imagenet_h100_g8_relu",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_uncased",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_cased",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_uncased_768",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_cased_768",
]


SUB_ALL = [
    "mutan_att_train_imagenet_relu_bert_cased_768",
    "mutan_att_train_imagenet_relu_bert_uncased_768",
    "mutan_att_train_imagenet_relu_bert_cased",
    "mutan_att_train_imagenet_relu_bert_uncased",
    "mutan_att_train_imagenet_relu",
    "mlb_att_train_imagenet_h200_g4_relu",
    "mlb_att_train_imagenet_h200_g4_relu_bert_uncased",
    "mlb_att_train_imagenet_h200_g4_relu_bert_cased",
    "mlb_att_train_imagenet_h200_g4_relu_bert_uncased_768",
    "mlb_att_train_imagenet_h200_g4_relu_bert_cased_768",
    "mlb_att_train_imagenet_h100_g8_relu",
    "mlb_att_train_imagenet_h100_g8_relu_bert_uncased",
    "mlb_att_train_imagenet_h100_g8_relu_bert_cased",
    "mlb_att_train_imagenet_h100_g8_relu_bert_uncased_768",
    "mlb_att_train_imagenet_h100_g8_relu_bert_cased_768",
    "globalbilinear_att_train_imagenet_h200_g4_relu",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_uncased",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_cased",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_uncased_768",
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_cased_768",
    "globalbilinear_att_train_imagenet_h100_g8_relu",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_uncased",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_cased",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_uncased_768",
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_cased_768",
    "globalbilinear_att_train_imagenet_h64_g8_relu",
]

SUB_BEST = [
    "globalbilinear_att_train_imagenet_h64_g8_relu",
]


DICT_METHOD = {
    "globalbilinear": SUB_GLOBAL_BILINEAR,
    "skipthoughts": SUB_SKIP_THOUGHTS,
    "bert3072": SUB_BERT_3072,
    "bert768": SUB_BERT_768,
    "all": SUB_ALL,
    "best": SUB_BEST
}


DICT_SCORE = {
    "mlb_att_train_imagenet_h200_g4_relu": 59.9,
    # "mlb_att_train_imagenet_h200_g4_relu_bert_uncased": 59.15,
    # "mlb_att_train_imagenet_h200_g4_relu_bert_cased": 58.45,
    # "mlb_att_train_imagenet_h200_g4_relu_bert_uncased_768": 59.45,
    # "mlb_att_train_imagenet_h200_g4_relu_bert_cased_768": 57.57,
    "mlb_att_train_imagenet_h100_g8_relu": 60.02,
    # "mlb_att_train_imagenet_h100_g8_relu_bert_uncased": 58.74,
    # "mlb_att_train_imagenet_h100_g8_relu_bert_cased": 58.73,
    # "mlb_att_train_imagenet_h100_g8_relu_bert_uncased_768": 60.09,
    # "mlb_att_train_imagenet_h100_g8_relu_bert_cased_768": 58.56,
    "globalbilinear_att_train_imagenet_h200_g4_relu": 59.62,
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_uncased": 58.83,
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_cased": 58.97,
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_uncased_768": 59.72,
    "globalbilinear_att_train_imagenet_h200_g4_relu_bert_cased_768": 59.12,
    "globalbilinear_att_train_imagenet_h100_g8_relu": 59.85,
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_uncased": 59.1,
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_cased": 59.33,
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_uncased_768": 60.09,
    "globalbilinear_att_train_imagenet_h100_g8_relu_bert_cased_768": 59.62,
    "globalbilinear_att_train_imagenet_h64_g8_relu": 60.12,
    # "globalbilinear_att_train_imagenet_h100_g8": 60.5,
    # "mutan_att_train_imagenet_relu_bert_cased_768": 57.75,
    # "mutan_att_train_imagenet_relu_bert_uncased_768": 58.35,
    # "mutan_att_train_imagenet_relu_bert_cased": 58.42,
    # "mutan_att_train_imagenet_relu_bert_uncased": 58.88,
    "mutan_att_train_imagenet_relu": 59.64,
}


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features.pickle"
BASE_EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features_base.pickle"
CASED_EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features_cased.pickle"

SUB_DIR = PROJECT_DIR + "/data/vqa_med/submission/"
path_utils.make_dir(SUB_DIR)
TEST_DIR = PROJECT_DIR + \
    "/data/raw/vqa_med/VQAMed2019Test/VQAMed2019_Test_ImageList.txt"

PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"    


def compute_prob_one_model(model_name, vqa_trainsplit="train"):

    parser = argparse.ArgumentParser(
        description='Train/Evaluate models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ##################################################
    # yaml options file contains all default choices #
    # parser.add_argument('--path_opt', default='options/breast/default.yaml', type=str,
    #                     help='path to a yaml options file')
    parser.add_argument('--path_opt', default='options/med/bilinear_att_train_imagenet_h200_g4.yaml', type=str,
                        help='path to a yaml options file')
    ################################################
    # change cli options to modify default choices #
    # logs options
    parser.add_argument('--dir_logs',
                        default='logs/med/train/globalbilinear_att_train_imagenet_h200_g4',
                        type=str, help='dir logs')
    # data options
    parser.add_argument('--vqa_trainsplit', type=str,
                        choices=['train', 'trainval'], default=vqa_trainsplit)
    # model options
    parser.add_argument('--arch', choices=models.model_names,
                        help='vqa model architecture: ' +
                        ' | '.join(models.model_names))
    parser.add_argument('--st_type',
                        help='skipthoughts type')
    parser.add_argument('--st_dropout', type=float)
    parser.add_argument('--st_fixed_emb', default=None, type=utils.str2bool,
                        help='backprop on embedding')
    # bert options
    parser.add_argument('--bert_model', default="bert-base-multilingual-uncased",
                        help='bert model: bert-base-uncased | bert-base-multilingual-uncased | bert-base-multilingual-cased')
    # image options
    parser.add_argument('--is_augment_image', default='1',
                        help='whether to augment images at the beginning of every epoch?')
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
    parser.add_argument('--resume', default='best', type=str,
                        help='path to latest checkpoint')
    parser.add_argument('--save_model', default=True, type=utils.str2bool,
                        help='able or disable save model and optim state')
    parser.add_argument('--save_all_from', type=int,
                        help='''delete the preceding checkpoint until an epoch,'''
                        ''' then keep all (useful to save disk space)')''')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation and test set', default=True)
    parser.add_argument('-j', '--workers', default=0, type=int,
                        help='number of data loading workers')
    parser.add_argument('--print_freq', '-p', default=10, type=int,
                        help='print frequency')
    ################################################
    parser.add_argument('-ho', '--help_opt', dest='help_opt', action='store_true',
                        help='show selected options before running')
    args = parser.parse_args()

    if vqa_trainsplit == "train":
        args.dir_logs = "logs/med/train/{}".format(model_name)
    else:
        args.dir_logs = "logs/med/trainval/{}".format(model_name)
    if "globalbilinear" in model_name:
        path_opt = model_name.replace("globalbilinear", "bilinear")
        if "_cased" in path_opt:
            path_opt = path_opt.replace("_cased", "")
        if "_uncased" in path_opt:
            path_opt = path_opt.replace("_uncased", "")
    elif "_cased_768" in model_name:
        path_opt = model_name.replace("_cased_768", "_768")
    elif "_uncased_768" in model_name:
        path_opt = model_name.replace("_uncased_768", "_768")
    elif "_cased" in model_name and "768" not in model_name:
        path_opt = model_name.replace("_cased", "")
    elif "_uncased" in model_name and "768" not in model_name:
        path_opt = model_name.replace("_uncased", "")
    else:
        path_opt = model_name
    path_opt = path_opt.replace("_trainval_", "_train_")
    args.path_opt = "{}/{}.yaml".format(args.dir_logs, path_opt)

    #########################################################################################
    # Create options
    #########################################################################################
    if "_cased" in model_name:
        args.bert_model = "bert-base-multilingual-cased"
    elif "_uncased" in model_name:
        args.bert_model = "bert-base-multilingual-uncased"

    if args.bert_model == "bert-base-uncased":
        question_features_path = BASE_EXTRACTED_QUES_FEATURES_PATH
    elif args.bert_model == "bert-base-multilingual-cased":
        question_features_path = CASED_EXTRACTED_QUES_FEATURES_PATH
    else:
        question_features_path = EXTRACTED_QUES_FEATURES_PATH

    options = {
        'vqa': {
            'trainsplit': args.vqa_trainsplit
        },
        'logs': {
            'dir_logs': args.dir_logs
        },
        'model': {
            'arch': args.arch,
            'seq2vec': {
                'type': args.st_type,
                'dropout': args.st_dropout,
                'fixed_emb': args.st_fixed_emb
            }
        },
        'optim': {
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'epochs': args.epochs
        }
    }
    if args.path_opt is not None:
        with open(args.path_opt, 'r') as handle:
            options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
        options = utils.update_values(options, options_yaml)
    print('## args')
    pprint(vars(args))
    print('## options')
    pprint(options)
    if args.help_opt:
        return

    # Set datasets options
    if 'vgenome' not in options:
        options['vgenome'] = None

    #########################################################################################
    # Create needed datasets
    #########################################################################################

    trainset = datasets.factory_VQA(options['vqa']['trainsplit'],
                                    options['vqa'],
                                    options['coco'],
                                    options['vgenome'])
    train_loader = trainset.data_loader(batch_size=options['optim']['batch_size'],
                                        num_workers=args.workers,
                                        shuffle=True)

    if options['vqa']['trainsplit'] == 'train':
        valset = datasets.factory_VQA('val', options['vqa'], options['coco'])
        val_loader = valset.data_loader(batch_size=options['optim']['batch_size'],
                                        num_workers=args.workers)

    if options['vqa']['trainsplit'] == 'trainval' or args.evaluate:
        testset = datasets.factory_VQA('test', options['vqa'], options['coco'])
        test_loader = testset.data_loader(batch_size=options['optim']['batch_size'],
                                          num_workers=args.workers)

    #########################################################################################
    # Create model, criterion and optimizer
    #########################################################################################

    model = models.factory(options['model'],
                           trainset.vocab_words(), trainset.vocab_answers(),
                           cuda=True, data_parallel=True)
    criterion = criterions.factory(options['vqa'], cuda=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 options['optim']['lr'])

    #########################################################################################
    # args.resume: resume from a checkpoint OR create logs directory
    #########################################################################################

    exp_logger = None
    if args.resume:
        args.start_epoch, best_acc1, exp_logger = load_checkpoint(model.module, optimizer,
                                                                  os.path.join(options['logs']['dir_logs'], args.resume))
    else:
        # Or create logs directory
        if os.path.isdir(options['logs']['dir_logs']):
            if click.confirm('Logs directory already exists in {}. Erase?'
                             .format(options['logs']['dir_logs'], default=False)):
                os.system('rm -r ' + options['logs']['dir_logs'])
            else:
                return
        os.system('mkdir -p ' + options['logs']['dir_logs'])
        path_new_opt = os.path.join(options['logs']['dir_logs'],
                                    os.path.basename(args.path_opt))
        path_args = os.path.join(options['logs']['dir_logs'], 'args.yaml')
        with open(path_new_opt, 'w') as f:
            yaml.dump(options, f, default_flow_style=False)
        with open(path_args, 'w') as f:
            yaml.dump(vars(args), f, default_flow_style=False)

    if exp_logger is None:
        # Set loggers
        exp_name = os.path.basename(
            options['logs']['dir_logs'])  # add timestamp
        exp_logger = logger.Experiment(exp_name, options)
        exp_logger.add_meters('train', make_meters())
        exp_logger.add_meters('test', make_meters())
        if options['vqa']['trainsplit'] == 'train':
            exp_logger.add_meters('val', make_meters())
        exp_logger.info['model_params'] = utils.params_count(model)
        print('Model has {} parameters'.format(
            exp_logger.info['model_params']))

    #########################################################################################
    # args.evaluate: on valset OR/AND on testset
    #########################################################################################

    if args.evaluate:
        path_logger_json = os.path.join(
            options['logs']['dir_logs'], 'logger.json')

        if options['vqa']['trainsplit'] == 'train':
            acc1, val_results, prob = engine.validate(val_loader, model, criterion,
                                                      exp_logger, args.start_epoch,
                                                      args.print_freq,
                                                      dict=io_utils.read_pickle(
                                                          question_features_path),
                                                      bert_dim=options["model"]["dim_q"],
                                                      is_return_prob=True)
        else:
            test_results, testdev_results, prob = engine.test(test_loader, model, exp_logger,
                                                              1, args.print_freq,
                                                              dict=io_utils.read_pickle(
                                                                  question_features_path),
                                                              bert_dim=options["model"]["dim_q"],
                                                              is_return_prob=True)

        torch.cuda.empty_cache()

        if vqa_trainsplit == "train":
            return prob, val_loader
        else:
            return prob, test_loader


# # method: avg, weighted, top predcition
def ensemble(dict_prob, val_loader, dict_runs, save_path, method="avg", vqa_trainsplit="train"):
    if method == "avg":
        i = 0
        for key in dict_runs:
            value = dict_prob[key]
            i += 1
            if i == 1:
                prob = value
            else:
                prob += value
        prob /= len(dict_runs)

    else:
        i = 0
        sum_weight = 0
        for key in dict_runs:
            value = dict_prob[key]
            i += 1
            weight = DICT_SCORE[key]
            if i == 1:
                prob = weight*value
            else:
                prob += weight*value
            sum_weight += weight
        prob /= sum_weight

    results = []
    pred_dict = {}
    gt_dict = {}
    acc = 0
    count = 0
    for i, sample in enumerate(val_loader):
        batch_size = len(sample["question_id"])
        prob_qi = prob[count:count+batch_size, :]

        if vqa_trainsplit == "train":
            target_answer = sample['answer'].data.cpu()
        for j in range(len(sample["question_id"])):
            values, indices = prob_qi[j].max(0)
            item = {'question_id': sample['question_id'][j],
                    'answer': val_loader.dataset.aid_to_ans[indices]}
            results.append(item)

            pred_dict[sample['question_id'][j]
                      ] = val_loader.dataset.aid_to_ans[indices]

            if vqa_trainsplit == "train":
                try:
                    gt_dict[sample['question_id'][j]
                            ] = val_loader.dataset.aid_to_ans[target_answer[j]]
                except:
                    gt_dict[sample['question_id'][j]
                            ] = val_loader.dataset.aid_to_ans[0]

                if pred_dict[sample['question_id'][j]] == gt_dict[sample['question_id'][j]]:
                    acc += 1

        count += batch_size

    if vqa_trainsplit == "train":
        acc = round(acc/count*100, 2)
        bleu = round(metrics_utils.compute_bleu_score(
            pred_dict, gt_dict)*100, 2)

    with open(save_path, 'w') as handle:
        json.dump(results, handle)

    if vqa_trainsplit == "train":
        return acc, bleu
    else:
        return results


def get_info(df, file_id):
    index = df.index[df['file_id'] == file_id].tolist()
    row = df.iloc[index[0]]
    return row


def main():
    dict_prob = {}
    df = pd.read_csv(PROCESSED_QA_PER_QUESTION_PATH)
    # for key, value in DICT_SCORE.items():
    #     prob, val_loader = compute_prob_one_model(model_name=key)
    #     dict_prob[key] = prob.detach()

    # for method in ["avg", "weighted"]:
    #     for ensem in ["best", "globalbilinear", "skipthoughts", "bert3072", "bert768", "all"]:
    #         sub_path = "{}ensemble/valid/{}_{}.json".format(
    #             SUB_DIR, ensem, method)
    #         sub = DICT_METHOD[ensem]
    #         acc, bleu = ensemble(dict_prob, val_loader,
    #                              sub, sub_path, method=method,
    #                              vqa_trainsplit="train")

    #         print(method, ensem, acc, bleu)

    for key, value in DICT_SCORE.items():
        prob, test_loader = compute_prob_one_model(model_name=key.replace(
            "_train_", "_trainval_"), vqa_trainsplit="trainval")
        dict_prob[key] = prob.detach()
        # del prob, test_loader

    for method in ["avg", "weighted"]:
        for ensem in ["globalbilinear", "skipthoughts"]:
            sub_path = "{}ensemble/test/{}_{}.txt".format(
                SUB_DIR, ensem, method)
            path_utils.make_dir("{}ensemble/test".format(SUB_DIR))
            sub = DICT_METHOD[ensem]
            results = ensemble(dict_prob, test_loader,
                               sub, sub_path, method=method,
                               vqa_trainsplit="trainval")

            with open(TEST_DIR, encoding='UTF-8') as f:
                lines = f.readlines()

            f = open(sub_path, 'w')
            count = 0
            for line in lines:
                file_id = line.split('\n')[0]
                row_info = get_info(df, file_id + ".jpg")
                question_id = row_info["question_id"] 
                f.write('{}|{}\n'.format(file_id, results[count]["answer"]))
                count += 1
            f.close()  # you can omit in most cases as the destructor will call it





def make_meters():
    meters_dict = {
        'loss': logger.AvgMeter(),
        'acc1': logger.AvgMeter(),
        'acc2': logger.AvgMeter(),
        'batch_time': logger.AvgMeter(),
        'data_time': logger.AvgMeter(),
        'epoch_time': logger.SumMeter()
    }
    return meters_dict


def save_results(results, epoch, split_name, dir_logs, dir_vqa):
    dir_epoch = os.path.join(dir_logs, 'epoch_' + str(epoch))
    name_json = 'OpenEnded_mscoco_{}_model_results.json'.format(split_name)
    # TODO: simplify formating
    if 'test' in split_name:
        name_json = 'vqa_' + name_json
    path_rslt = os.path.join(dir_epoch, name_json)
    os.system('mkdir -p ' + dir_epoch)
    with open(path_rslt, 'w') as handle:
        json.dump(results, handle)
    # if not 'test' in split_name:
    #     os.system('python main/eval_res.py --dir_vqa {} --dir_epoch {} --subtype {} &'
    #               .format(dir_vqa, dir_epoch, split_name))


def save_checkpoint(info, model, optim, dir_logs, save_model, save_all_from=None, is_best=True):
    os.system('mkdir -p ' + dir_logs)
    if save_all_from is None:
        path_ckpt_info = os.path.join(dir_logs, 'ckpt_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_model.pth.tar')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_optim.pth.tar')
        path_best_info = os.path.join(dir_logs, 'best_info.pth.tar')
        path_best_model = os.path.join(dir_logs, 'best_model.pth.tar')
        path_best_optim = os.path.join(dir_logs, 'best_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info)
        if is_best:
            shutil.copyfile(path_ckpt_info, path_best_info)
        # save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model)
            torch.save(optim, path_ckpt_optim)
            if is_best:
                shutil.copyfile(path_ckpt_model, path_best_model)
                shutil.copyfile(path_ckpt_optim, path_best_optim)
    else:
        is_best = False  # because we don't know the test accuracy
        path_ckpt_info = os.path.join(dir_logs, 'ckpt_epoch,{}_info.pth.tar')
        path_ckpt_model = os.path.join(dir_logs, 'ckpt_epoch,{}_model.pth.tar')
        path_ckpt_optim = os.path.join(dir_logs, 'ckpt_epoch,{}_optim.pth.tar')
        # save info & logger
        path_logger = os.path.join(dir_logs, 'logger.json')
        info['exp_logger'].to_json(path_logger)
        torch.save(info, path_ckpt_info.format(info['epoch']))
        # save model state & optim state
        if save_model:
            torch.save(model, path_ckpt_model.format(info['epoch']))
            torch.save(optim, path_ckpt_optim.format(info['epoch']))
        if info['epoch'] > 1 and info['epoch'] < save_all_from + 1:
            os.system('rm ' + path_ckpt_info.format(info['epoch'] - 1))
            os.system('rm ' + path_ckpt_model.format(info['epoch'] - 1))
            os.system('rm ' + path_ckpt_optim.format(info['epoch'] - 1))
    if not save_model:
        print('Warning train.py: checkpoint not saved')


def load_checkpoint(model, optimizer, path_ckpt):
    path_ckpt_info = path_ckpt + '_info.pth.tar'
    path_ckpt_model = path_ckpt + '_model.pth.tar'
    path_ckpt_optim = path_ckpt + '_optim.pth.tar'
    print('---------------------------------------------')
    print(path_ckpt_info)
    print(path_ckpt_model)
    print(path_ckpt_optim)
    print('---------------------------------------------')
    if os.path.isfile(path_ckpt_info):
        info = torch.load(path_ckpt_info)
        start_epoch = 0
        best_acc1 = 0
        exp_logger = None
        if 'epoch' in info:
            start_epoch = info['epoch']
        else:
            print('Warning train.py: no epoch to resume')
        if 'best_acc1' in info:
            best_acc1 = info['best_acc1']
        else:
            print('Warning train.py: no best_acc1 to resume')
        if 'exp_logger' in info:
            exp_logger = info['exp_logger']
        else:
            print('Warning train.py: no exp_logger to resume')
    else:
        print("Warning train.py: no info checkpoint found at '{}'".format(
            path_ckpt_info))
    if os.path.isfile(path_ckpt_model):
        model_state = torch.load(path_ckpt_model)
        model.load_state_dict(model_state)
    else:
        print("Warning train.py: no model checkpoint found at '{}'".format(
            path_ckpt_model))
    if optimizer is not None and os.path.isfile(path_ckpt_optim):
        optim_state = torch.load(path_ckpt_optim)
        optimizer.load_state_dict(optim_state)
    else:
        print("Warning train.py: no optim checkpoint found at '{}'".format(
            path_ckpt_optim))
    print("=> loaded checkpoint '{}' (epoch {}, best_acc1 {})"
          .format(path_ckpt, start_epoch, best_acc1))
    return start_epoch, best_acc1, exp_logger


if __name__ == '__main__':
    main()
