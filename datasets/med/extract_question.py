import pandas as pd
from vqa.models import sen2vec
import os
import datasets.utils.paths_utils as path_utils
import datasets.utils.io_utils as io_utils
import datasets.utils.print_utils as print_utils
from vqa.models import sen2vec


CURRENT_WORKING_DIR = os.path.realpath(__file__)
PROJECT_DIR = path_utils.get_project_dir(CURRENT_WORKING_DIR, "vqa_idrid")
RAW_DIR = PROJECT_DIR + "/data/vqa_med/raw/raw/"
PROCESSED_QA_PER_QUESTION_PATH = RAW_DIR + "med_qa_per_question.csv"
EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features.pickle"
BASE_EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features_base.pickle"
CASED_EXTRACTED_QUES_FEATURES_PATH = RAW_DIR + "question_features_cased.pickle"


def main():

    df = pd.read_csv(PROCESSED_QA_PER_QUESTION_PATH)
    list_questions = list(df.question)
    list_questions = list(set(list_questions))

    #########################################################################################
    # Begin training on train/val or trainval/test
    #########################################################################################

    dict = {}
    batch = 4
    for i in range(0, len(list_questions), batch):
        fr, to = i, i+batch
        if to >= len(list_questions):
            to = len(list_questions)
        print_utils.print_tqdm(to, len(list_questions), cutoff=2)
        questions = list_questions[fr:to]
        input_question = sen2vec.sen2vec(questions, bert_model="bert-base-multilingual-cased")
        for q, question in enumerate(questions):
            dict[question] = input_question[q].cpu().detach().numpy()
        # print(len(dict))
        del questions, input_question
        # torch.cuda.empty_cache()

    io_utils.write_pickle(dict, CASED_EXTRACTED_QUES_FEATURES_PATH)


if __name__ == '__main__':
    main()
