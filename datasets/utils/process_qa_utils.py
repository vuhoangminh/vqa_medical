import sys
import json
import re


def get_top_answers(examples, nans=3):
    counts = {}
    for ex in examples:
        ans = ex['answer']
        counts[ans] = counts.get(ans, 0) + 1

    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('Top answer and their counts:')
    print('\n'.join(map(str, cw[:])))

    vocab = []
    for i in range(nans):
        vocab.append(cw[i][1])
    return vocab[:nans]


def remove_examples(examples, ans_to_aid):
    new_examples = []
    for i, ex in enumerate(examples):
        if ex['answer'] in ans_to_aid:
            new_examples.append(ex)
    print('Number of examples reduced from %d to %d ' %
          (len(examples), len(new_examples)))
    return new_examples


def tokenize(sentence):
    return [i for i in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", sentence) if i != '' and i != ' ' and i != '\n']


def tokenize_mcb(s):
    t_str = s.lower()
    for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\.', r'\;']:
        t_str = re.sub(i, '', t_str)
    for i in [r'\-', r'\/']:
        t_str = re.sub(i, ' ', t_str)
    q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list


def preprocess_questions(examples, nlp='nltk'):
    if nlp == 'nltk':
        from nltk.tokenize import word_tokenize
    print('Example of generated tokens after preprocessing some questions:')
    for i, ex in enumerate(examples):
        s = ex['question']
        if nlp == 'nltk':
            ex['question_words'] = word_tokenize(str(s).lower())
        elif nlp == 'mcb':
            ex['question_words'] = tokenize_mcb(s)
        else:
            ex['question_words'] = tokenize(s)
        if i < 10:
            print(ex['question_words'])
        if i % 1000 == 0:
            sys.stdout.write("processing %d/%d (%.2f%% done)   \r" %
                             (i, len(examples), i*100.0/len(examples)))
            sys.stdout.flush()
    return examples


def remove_long_tail_train(examples, minwcount=0):
    # Replace words which are in the long tail (counted less than 'minwcount' times) by the UNK token.
    # Also create vocab, a list of the final words.

    # count up the number of words
    counts = {}
    for ex in examples:
        for w in ex['question_words']:
            counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count, w) for w, count in counts.items()], reverse=True)
    print('Top words and their counts:')
    print('\n'.join(map(str, cw[:20])))

    total_words = sum(counts.values())
    print('Total words:', total_words)
    bad_words = [w for w, n in counts.items() if n <= minwcount]
    vocab = [w for w, n in counts.items() if n > minwcount]
    bad_count = sum(counts[w] for w in bad_words)
    print('Number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('Number of words in vocab would be %d' % (len(vocab), ))
    print('Number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count*100.0/total_words))

    print('Insert the special UNK token')
    vocab.append('UNK')
    for ex in examples:
        words = ex['question_words']
        question = [w if counts.get(
            w, 0) > minwcount else 'UNK' for w in words]
        ex['question_words_UNK'] = question

    return examples, vocab


def remove_long_tail_test(examples, word_to_wid):
    for ex in examples:
        ex['question_words_UNK'] = [
            w if w in word_to_wid else 'UNK' for w in ex['question_words']]
    return examples


def encode_question(examples, word_to_wid, maxlength=15, pad='left'):
    # Add to tuple question_wids and question_length
    for i, ex in enumerate(examples):
        # record the length of this sequence
        ex['question_length'] = min(maxlength, len(ex['question_words_UNK']))
        ex['question_wids'] = [0]*maxlength
        for k, w in enumerate(ex['question_words_UNK']):
            if k < maxlength:
                if pad == 'right':
                    ex['question_wids'][k] = word_to_wid[w]
                else:  # ['pad'] == 'left'
                    new_k = k + maxlength - len(ex['question_words_UNK'])
                    ex['question_wids'][new_k] = word_to_wid[w]
                ex['seq_length'] = len(ex['question_words_UNK'])
    return examples


def encode_answer(examples, ans_to_aid):
    print('Warning: aid of answer not in vocab is 1999')
    for i, ex in enumerate(examples):
        ex['answer_aid'] = ans_to_aid.get(
            ex['answer'], 1999)  # -1 means answer not in vocab
    return examples


def encode_answers_occurence(examples, ans_to_aid):
    for i, ex in enumerate(examples):
        answers = []
        answers_aid = []
        answers_count = []
        for ans in ex['answers_occurence']:
            aid = ans_to_aid.get(ans[0], -1)  # -1 means answer not in vocab
            if aid != -1:
                answers.append(ans[0])
                answers_aid.append(aid)
                answers_count.append(ans[1])
        ex['answers'] = answers
        ex['answers_aid'] = answers_aid
        ex['answers_count'] = answers_count
    return examples
