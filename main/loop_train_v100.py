import os

cmd_list = list()

list_options = ["breast/minhmul_att",
                "breast/minhmul_noatt",
                "tools/minhmul_att",
                "tools/minhmul_noatt",
                "idrid/minhmul_att",
                "idrid/minhmul_noatt",
                "vqa/minhmul_att",
                "vqa/minhmul_noatt",
                "vqa2/minhmul_att",
                "vqa2/minhmul_noatt",
                ]


for option in list_options:

    if "minhmul_att" in option and "breast" in option:
        cmd = "CUDA_VISIBLE_DEVICES=1 python main/train.py --path_opt options/{}_train_relu.yaml --dir_logs logs/{}_train_relu --vqa_trainsplit train -b 256 --resume ckpt".format(
            option, option
        )
    elif "minhmul_att" in option:
        cmd = "CUDA_VISIBLE_DEVICES=1 python main/train.py --path_opt options/{}_train_relu.yaml --dir_logs logs/{}_train_relu --vqa_trainsplit train -b 256".format(
            option, option
        )
    else:
        cmd = "CUDA_VISIBLE_DEVICES=1 python main/train.py --path_opt options/{}_train_relu.yaml --dir_logs logs/{}_train_relu --vqa_trainsplit train -b 512".format(
            option, option
        )

    try:
        print(">> RUNNING:", cmd)
        os.system(cmd)
        import torch
        torch.cuda.empty_cache()
    except:
        print("something wrong")