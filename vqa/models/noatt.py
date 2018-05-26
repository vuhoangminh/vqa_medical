import torch
import torch.nn as nn
import torch.nn.functional as F

from vqa.lib import utils
from vqa.models import fusion
from vqa.models import seq2vec


class AbstractNoAtt(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(AbstractNoAtt, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        # Modules
        self.seq2vec = seq2vec.factory(self.vocab_words, self.opt['seq2vec'])
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_h'], self.num_classes)

    def _fusion(self, input_v, input_q):
        raise NotImplementedError

    def _classif(self, x):
        if 'activation' in self.opt['classif']:
            x = getattr(F, self.opt['classif']['activation'])(x)
        x = F.dropout(x, p=self.opt['classif']['dropout'], training=self.training)
        x = self.linear_classif(x)
        return x

    def forward(self, input_v, input_q):
        # print ("input_q size is:", input_q.size())
        # print ("input_v size is:", input_v.size())
        # print ("fusion...")
        x_q = self.seq2vec(input_q)
        # print ("x_q size (seq2vec(input_q)) is:", x_q.size())
        x = self._fusion(input_v, x_q)
        # print ("x size (after fusion) is:", x.size())
        x = self._classif(x)
        # print ("x size (after classification) is:", x.size())
        return x


# class ElementdivNoAtt(AbstractNoAtt):

#     def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
#         super(ElementdivNoAtt, self).__init__(opt, vocab_words, vocab_answers)
#         self.fusion = fusion.ElementdivFusion(self.opt['fusion'])

#     def _fusion(self, input_v, input_q):
#         x = self.fusion(input_v, input_q)
#         return x


class MinhmulNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(MinhmulNoAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.MinhmulFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x


class MinhsumNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(MinhsumNoAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.MinhsumFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x


class ElementsumNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(ElementsumNoAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.ElementsumFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x


class ConcatNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(ConcatNoAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.ConcatFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x		

	
class MLBNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(MLBNoAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.MLBFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x


class MutanNoAtt(AbstractNoAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        opt['fusion']['dim_h'] = opt['fusion']['dim_mm']
        super(MutanNoAtt, self).__init__(opt, vocab_words, vocab_answers)
        self.fusion = fusion.MutanFusion(self.opt['fusion'])

    def _fusion(self, input_v, input_q):
        x = self.fusion(input_v, input_q)
        return x

