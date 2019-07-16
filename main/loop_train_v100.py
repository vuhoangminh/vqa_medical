import os
import random

def run_loop():

    cmd_list = list()
    cmd_resume_list = list()
    logs_list = list()

    list_options = [
        "minhmul_att_train_selu_h200_g8",
        "minhmul_att_train_relu_h200_g8",
        "minhmul_att_train_selu_h200_g4",
        "minhmul_att_train_relu_h200_g4",    
        "minhmul_att_train_selu",
        "minhmul_att_train_relu",
    ]

    list_dataset = [
        "breast",
        "idrid",
        "tools",
        # "vqa",
        # "vqa2",        
    ]

    for dataset in list_dataset:
        for option in list_options:
            logs = "logs/{}/{}".format(dataset, option)
            cmd = "python main/train.py --path_opt options/{}/{}.yaml --dir_logs {} --vqa_trainsplit train -b 256 --epochs 120".format(dataset, option, logs)
            cmd_resume = "python main/train.py --path_opt options/{}/{}.yaml --dir_logs {} --vqa_trainsplit train -b 256 --epochs 120 --resume ckpt".format(dataset, option, logs)
            cmd_list.append(cmd)
            cmd_resume_list(cmd_resume)
            logs_list.append(logs)

    combined = list(zip(logs_list, cmd_list, cmd_resume_list))
    random.shuffle(combined)

    logs_list, cmd_list, cmd_resume_list = zip(*combined)

    # pp.pprint(combined)

    for i in range(len(cmd_list)):
        cmd = cmd_list[i]
        logs = logs_list[i]

        if not os.path.exists(logs):
            try:
                print("========================================================================================================")
                print(">> RUNNING:", cmd)
                print("========================================================================================================")
                os.system(cmd)
                import torch
                torch.cuda.empty_cache()
            except:
                print("something wrong")
        else:
            try:
                print("========================================================================================================")
                print(">> RUNNING:", cmd)
                print("========================================================================================================")
                os.system(cmd)
                import torch
                torch.cuda.empty_cache()
            except:
                print("something wrong")            


def main():
    run_loop()


if __name__ == "__main__":
    main()
