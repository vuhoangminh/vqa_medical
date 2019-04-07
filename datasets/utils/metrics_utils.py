import os
import sys
import argparse
import string
import csv
import nltk
import warnings

from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import datasets.utils.paths_utils as path_utils
import datasets.utils.image_utils as image_utils


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
QA_VALID_TXT = PROJECT_DIR + \
    "/data/raw/vqa_med/ImageClef-2019-VQA-Med-Validation/All_QA_Pairs_val.txt"


def compute_bleu_score(candidate_pairs, gt_pairs,
                       case_sensitive=False,
                       remove_stopwords=False,
                       stemming=False,
                       verbose=False):
    translator = str.maketrans('', '', string.punctuation)
    warnings.filterwarnings('ignore')

    # Stats on the captions
    min_words = sys.maxsize
    max_words = 0
    max_sent = 0
    total_words = 0
    words_distrib = {}

    # NLTK
    # Download Punkt tokenizer (for word_tokenize method)
    # Download stopwords (for stopword removal)
    nltk.download('punkt')
    nltk.download('stopwords')

    # English Stopwords
    stops = set(stopwords.words("english"))

    # Stemming
    stemmer = SnowballStemmer("english")

    # Define max score and current score
    max_score = len(gt_pairs)
    current_score = 0

    i = 0
    for image_key in candidate_pairs:

        # Get candidate and GT caption
        candidate_caption = candidate_pairs[image_key]
        gt_caption = gt_pairs[image_key]

        # Optional - Go to lowercase
        if not case_sensitive:
            candidate_caption = candidate_caption.lower()
            gt_caption = gt_caption.lower()

        # Split caption into individual words (remove punctuation)
        candidate_words = nltk.tokenize.word_tokenize(
            candidate_caption.translate(translator))
        gt_words = nltk.tokenize.word_tokenize(
            gt_caption.translate(translator))

        # Corpus stats
        total_words += len(gt_words)
        gt_sentences = nltk.tokenize.sent_tokenize(gt_caption)

        # Optional - Remove stopwords
        if remove_stopwords:
            candidate_words = [
                word for word in candidate_words if word.lower() not in stops]
            gt_words = [word for word in gt_words if word.lower() not in stops]

        # Optional - Apply stemming
        if stemming:
            candidate_words = [stemmer.stem(word) for word in candidate_words]
            gt_words = [stemmer.stem(word) for word in gt_words]

        # Calculate BLEU score for the current caption
        try:
            # If both the GT and candidate are empty, assign a score of 1 for this caption
            if len(gt_words) == 0 and len(candidate_words) == 0:
                bleu_score = 1
            # Calculate the BLEU score
            else:
                bleu_score = nltk.translate.bleu_score.sentence_bleu(
                    [gt_words], candidate_words, smoothing_function=SmoothingFunction().method0)
        # Handle problematic cases where BLEU score calculation is impossible
        except ZeroDivisionError:
            print('Problem with ', gt_words, candidate_words)

        # Increase calculated score
        current_score += bleu_score
        nb_words = str(len(gt_words))
        if nb_words not in words_distrib:
            words_distrib[nb_words] = 1
        else:
            words_distrib[nb_words] += 1

        # Corpus stats
        if len(gt_words) > max_words:
            max_words = len(gt_words)

        if len(gt_words) < min_words:
            min_words = len(gt_words)

        if len(gt_sentences) > max_sent:
            max_sent = len(gt_sentences)

        # Progress display
        i += 1

        if verbose:
            if i % 1000 == 0:
                print(i, '/', len(gt_pairs), ' captions processed...')

    if verbose:
        # Print stats
        print('Corpus statistics\n********************************')
        print('Number of words distribution')
        print_dict_sorted_num(words_distrib)
        print('Least words in caption :', min_words)
        print('Most words in caption :', max_words)
        print('Average words in caption :', total_words / len(gt_pairs))
        print('Most sentences in caption :', max_sent)

        # Print evaluation result
        print('Final result\n********************************')
        print('Obtained score :', current_score, '/', max_score)
        print('Mean score over all captions :', current_score / max_score)

    return current_score / max_score


def main():

    # Hide warnings
    warnings.filterwarnings('ignore')

    # Stats on the captions
    min_words = sys.maxsize
    max_words = 0
    max_sent = 0
    total_words = 0
    words_distrib = {}

    # NLTK
    # Download Punkt tokenizer (for word_tokenize method)
    # Download stopwords (for stopword removal)
    nltk.download('punkt')
    nltk.download('stopwords')

    # English Stopwords
    stops = set(stopwords.words("english"))

    # Stemming
    stemmer = SnowballStemmer("english")

    # Remove punctuation from string
    translator = str.maketrans('', '', string.punctuation)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate_file',
                        default=QA_VALID_TXT,
                        help='path to the candidate file to evaluate')
    parser.add_argument('--gt_file', default=QA_VALID_TXT,
                        help='path to the ground truth file')
    parser.add_argument('-r', '--remove-stopwords', default=False,
                        action='store_true', help='enable stopword removal')
    parser.add_argument('-s', '--stemming', default=False,
                        action='store_true', help='enable stemming')
    parser.add_argument('-c', '--case-sensitive', default=False,
                        action='store_true', help='case-sensitive evaluation')
    args = parser.parse_args()

    # Read files
    print('Input parameters\n********************************')

    print('Candidate file is "' + args.candidate_file + '"')
    candidate_pairs = readfile(args.candidate_file)

    print('Ground Truth file is "' + args.gt_file + '"')
    gt_pairs = readfile(args.gt_file)

    print('Removing stopwords is "' + str(args.remove_stopwords) + '"')
    print('Stemming is "' + str(args.stemming) + '"')

    # Define max score and current score
    max_score = len(gt_pairs)
    current_score = 0

    # Evaluate each candidate caption against the ground truth
    print('Processing captions...\n********************************')

    i = 0
    for image_key in candidate_pairs:

        # Get candidate and GT caption
        candidate_caption = candidate_pairs[image_key]
        gt_caption = gt_pairs[image_key]

        # Optional - Go to lowercase
        if not args.case_sensitive:
            candidate_caption = candidate_caption.lower()
            gt_caption = gt_caption.lower()

        # Split caption into individual words (remove punctuation)
        candidate_words = nltk.tokenize.word_tokenize(
            candidate_caption.translate(translator))
        gt_words = nltk.tokenize.word_tokenize(
            gt_caption.translate(translator))

        # Corpus stats
        total_words += len(gt_words)
        gt_sentences = nltk.tokenize.sent_tokenize(gt_caption)

        # Optional - Remove stopwords
        if args.remove_stopwords:
            candidate_words = [
                word for word in candidate_words if word.lower() not in stops]
            gt_words = [word for word in gt_words if word.lower() not in stops]

        # Optional - Apply stemming
        if args.stemming:
            candidate_words = [stemmer.stem(word) for word in candidate_words]
            gt_words = [stemmer.stem(word) for word in gt_words]

        # Calculate BLEU score for the current caption
        try:
            # If both the GT and candidate are empty, assign a score of 1 for this caption
            if len(gt_words) == 0 and len(candidate_words) == 0:
                bleu_score = 1
            # Calculate the BLEU score
            else:
                bleu_score = nltk.translate.bleu_score.sentence_bleu(
                    [gt_words], candidate_words, smoothing_function=SmoothingFunction().method0)
        # Handle problematic cases where BLEU score calculation is impossible
        except ZeroDivisionError:
            print('Problem with ', gt_words, candidate_words)

        # Increase calculated score
        current_score += bleu_score
        nb_words = str(len(gt_words))
        if nb_words not in words_distrib:
            words_distrib[nb_words] = 1
        else:
            words_distrib[nb_words] += 1

        # Corpus stats
        if len(gt_words) > max_words:
            max_words = len(gt_words)

        if len(gt_words) < min_words:
            min_words = len(gt_words)

        if len(gt_sentences) > max_sent:
            max_sent = len(gt_sentences)

        # Progress display
        i += 1
        if i % 1000 == 0:
            print(i, '/', len(gt_pairs), ' captions processed...')

    # Print stats
    print('Corpus statistics\n********************************')
    print('Number of words distribution')
    print_dict_sorted_num(words_distrib)
    print('Least words in caption :', min_words)
    print('Most words in caption :', max_words)
    print('Average words in caption :', total_words / len(gt_pairs))
    print('Most sentences in caption :', max_sent)

    # Print evaluation result
    print('Final result\n********************************')
    print('Obtained score :', current_score, '/', max_score)
    print('Mean score over all captions :', current_score / max_score)


# Read a Tab-separated ImageID - Caption pair file
def readfile(path):
    try:
        pairs = {}
        with open(path, encoding='UTF-8') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split("|")
            print(line)
            image, question, answer = line[0], line[1], line[2].split("\n")[
                0]
            pairs[line[0]] = answer

        return pairs
    except FileNotFoundError:
        print('File "' + path + '" not found! Please check the path!')
        exit(1)


# Print 1-level key-value dictionary, sorted (with numeric key)
def print_dict_sorted_num(obj):
    keylist = [int(x) for x in list(obj.keys())]
    keylist.sort()
    for key in keylist:
        print(key, ':', obj[str(key)])


# Main
if __name__ == '__main__':
    main()
