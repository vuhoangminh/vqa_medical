import sys
import copy
import torch
import torch.nn as nn
import torchvision.models as models

# from .noatt import ConcatNoAtt, ElementsumNoAtt, MLBNoAtt, MutanNoAtt 
# from .att import ConcatAtt, ElementsumAtt, MLBAtt, MutanAtt

from .noatt import ElementsumNoAtt, MLBNoAtt, MutanNoAtt, MinhsumNoAtt, MinhmulNoAtt 
# from .att import MinhsumAtt, MinhmulAtt, ElementsumAtt, MLBAtt, MutanAtt, BilinearAtt
from .att import MinhmulAtt, BilinearAtt, MLBAtt, MutanAtt

# from .noatt import MLBNoAtt, MutanNoAtt, ConcatNoAtt, ElementsumNoAtt
# from .att import MLBAtt, MutanAtt, ConcatAtt, ElementsumAtt

model_names = sorted(name for name in sys.modules[__name__].__dict__
    if not name.startswith("__"))# and 'Att' in name)

def factory(opt, vocab_words, vocab_answers, cuda=True, data_parallel=True):
    opt = copy.copy(opt)

    if opt['arch'] in model_names:
        model = getattr(sys.modules[__name__], opt['arch'])(opt, vocab_words, vocab_answers)
    else:
        raise ValueError

    if data_parallel:
        model = nn.DataParallel(model).cuda()
        if not cuda:
            raise ValueError

    if cuda:
        model.cuda()

    return model