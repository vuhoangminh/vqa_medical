import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

from vqa.lib import utils
from vqa.models import seq2vec
from vqa.models import fusion


def return_self(input):
    return input


class AbstractAtt(nn.Module):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        super(AbstractAtt, self).__init__()
        self.opt = opt
        self.vocab_words = vocab_words
        self.vocab_answers = vocab_answers
        self.num_classes = len(self.vocab_answers)
        # Modules
        if self.opt['seq2vec']['arch'] == "bert":
            self.seq2vec = return_self
        else:
            self.seq2vec = seq2vec.factory(
                self.vocab_words, self.opt['seq2vec'])
        # Modules for attention
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'],
                                    self.opt['attention']['dim_v'], 1, 1)
        self.linear_q_att = nn.Linear(self.opt['dim_q'],
                                      self.opt['attention']['dim_q'])
        self.conv_att = nn.Conv2d(self.opt['attention']['dim_mm'],
                                  self.opt['attention']['nb_glimpses'], 1, 1)
        # Modules for batch norm
        self.batchnorm_conv_v_att = nn.BatchNorm2d(
            self.opt['attention']['dim_v'])
        self.batchnorm_linear_q_att = nn.BatchNorm1d(
            self.opt['attention']['dim_q'])
        self.batchnorm_conv_att = nn.BatchNorm2d(
            self.opt['attention']['nb_glimpses'])
        self.batchnorm_fusion_att = nn.BatchNorm1d(
            self.opt['attention']['dim_mm'])
        self.batchnorm_list_linear_v_fusion = nn.BatchNorm1d(
            self.opt['attention']['dim_mm'])
        self.batchnorm_list_linear_q_fusion = nn.BatchNorm1d(
            self.opt['attention']['dim_mm']*self.opt['attention']['nb_glimpses'])
        self.batchnorm_fusion_classif = nn.BatchNorm1d(
            self.opt['attention']['dim_mm']*self.opt['attention']['nb_glimpses'])

        # Modules for classification
        self.list_linear_v_fusion = None
        self.linear_q_fusion = None
        self.linear_classif = None

    def _fusion_att(self, x_v, x_q):
        raise NotImplementedError

    def _fusion_classif(self, x_v, x_q):
        raise NotImplementedError

    def _attention(self, input_v, x_q_vec):
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)

        # Process visual before fusion
        #x_v = input_v.view(batch_size*width*height, dim_features)
        x_v = input_v
        x_v = F.dropout(x_v,
                        p=self.opt['attention']['dropout_v'],
                        training=self.training)
        x_v = self.conv_v_att(x_v)
        # x_v = self.batchnorm_conv_v_att(x_v)
        if 'activation_v' in self.opt['attention']:
            x_v = getattr(F, self.opt['attention']['activation_v'])(x_v)
        x_v = x_v.view(batch_size,
                       self.opt['attention']['dim_v'],
                       width * height)
        x_v = x_v.transpose(1, 2)

        # Process question before fusion
        x_q = F.dropout(x_q_vec, p=self.opt['attention']['dropout_q'],
                        training=self.training)
        x_q = self.linear_q_att(x_q)
        # x_q = self.batchnorm_linear_q_att(x_q)
        if 'activation_q' in self.opt['attention']:
            x_q = getattr(F, self.opt['attention']['activation_q'])(x_q)
        x_q = x_q.view(batch_size,
                       1,
                       self.opt['attention']['dim_q'])
        x_q = x_q.expand(batch_size,
                         width * height,
                         self.opt['attention']['dim_q'])

        # First multimodal fusion
        x_att = self._fusion_att(x_v, x_q)
        x_att = x_att.transpose(1, 2)
        # x_att = self.batchnorm_fusion_att(x_att)
        x_att = x_att.transpose(1, 2)
        if 'activation_mm' in self.opt['attention']:
            x_att = getattr(F, self.opt['attention']['activation_mm'])(x_att)

        # Process attention vectors
        x_att = F.dropout(x_att,
                          p=self.opt['attention']['dropout_mm'],
                          training=self.training)
        # can be optim to avoid two views and transposes
        x_att = x_att.view(batch_size,
                           width,
                           height,
                           self.opt['attention']['dim_mm'])
        x_att = x_att.transpose(2, 3).transpose(1, 2)
        x_att = self.conv_att(x_att)
        # x_att = self.batchnorm_conv_att(x_att)
        x_att = x_att.view(batch_size,
                           self.opt['attention']['nb_glimpses'],
                           width * height)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width*height)
            x_att = F.softmax(x_att)
            list_att.append(x_att)

        self.list_att = [x_att.data for x_att in list_att]

        # Apply attention vectors to input_v
        x_v = input_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v = x_v.transpose(1, 2)

        list_v_att = []
        list_v_record = []
        for i, x_att in enumerate(list_att):
            x_att = x_att.view(batch_size,
                               width * height,
                               1)
            x_att = x_att.expand(batch_size,
                                 width * height,
                                 self.opt['dim_v'])
            x_v_att = torch.mul(x_att, x_v)
            list_v_record.append(x_v_att)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(batch_size, self.opt['dim_v'])
            list_v_att.append(x_v_att)

        return list_v_att, list_v_record

    def _fusion_glimpses(self, list_v_att, x_q_vec):
        # Process visual for each glimpses
        list_v = []
        for glimpse_id, x_v_att in enumerate(list_v_att):
            x_v = F.dropout(x_v_att,
                            p=self.opt['fusion']['dropout_v'],
                            training=self.training)
            x_v = self.list_linear_v_fusion[glimpse_id](x_v)
            # x_v = self.batchnorm_list_linear_v_fusion(x_v)
            if 'activation_v' in self.opt['fusion']:
                x_v = getattr(F, self.opt['fusion']['activation_v'])(x_v)
            list_v.append(x_v)
        x_v = torch.cat(list_v, 1)

        # Process question
        x_q = F.dropout(x_q_vec,
                        p=self.opt['fusion']['dropout_q'],
                        training=self.training)
        x_q = self.linear_q_fusion(x_q)
        # x_q = self.batchnorm_list_linear_q_fusion(x_q)
        if 'activation_q' in self.opt['fusion']:
            x_q = getattr(F, self.opt['fusion']['activation_q'])(x_q)

        # Second multimodal fusion
        x = self._fusion_classif(x_v, x_q)
        # x = self.batchnorm_fusion_classif(x)
        return x

    def _classif(self, x):

        if 'activation' in self.opt['classif']:
            x = getattr(F, self.opt['classif']['activation'])(x)
        x = F.dropout(x,
                      p=self.opt['classif']['dropout'],
                      training=self.training)
        x = self.linear_classif(x)
        return x

    def forward(self, input_v, input_q):

        # if not hasattr(self, 'seq2vec'):
        if input_v.dim() != 4 and input_q.dim() != 2:
            raise ValueError

        x_q_vec = self.seq2vec(input_q)
        list_v_att, list_v_record = self._attention(input_v, x_q_vec)
        x = self._fusion_glimpses(list_v_att, x_q_vec)
        x = self._classif(x)
        return x, list_v_record


class MinhmulAtt(AbstractAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        # TODO: deep copy ?
        opt['attention']['dim_v'] = opt['attention']['dim_h']
        opt['attention']['dim_q'] = opt['attention']['dim_h']
        opt['attention']['dim_mm'] = opt['attention']['dim_h']
        super(MinhmulAtt, self).__init__(opt, vocab_words, vocab_answers)
        # Modules for classification
        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(self.opt['dim_v'],
                      self.opt['fusion']['dim_h'])
            for i in range(self.opt['attention']['nb_glimpses'])])
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                         self.opt['fusion']['dim_h']
                                         * self.opt['attention']['nb_glimpses'])
        self.linear_classif = nn.Linear(self.opt['fusion']['dim_h']
                                        * self.opt['attention']['nb_glimpses'],
                                        self.num_classes)

    def _fusion_att(self, x_v, x_q):
        x_att = torch.pow(x_q, 2)
        x_att = torch.mul(x_v, x_att)
        return x_att

    def _fusion_classif(self, x_v, x_q):
        x_mm = torch.pow(x_q, 2)
        x_mm = torch.mul(x_v, x_mm)
        return x_mm


class BilinearAtt(AbstractAtt):

    def __init__(self, opt={}, vocab_words=[], vocab_answers=[]):
        # TODO: deep copy ?
        opt['attention']['dim_v'] = opt['attention']['dim_h']
        opt['attention']['dim_q'] = opt['attention']['dim_h']
        opt['attention']['dim_mm'] = opt['attention']['dim_h']
        super(BilinearAtt, self).__init__(opt, vocab_words, vocab_answers)
        # Modules for classification
        self.list_linear_v_fusion = nn.ModuleList([
            nn.Linear(self.opt['dim_v'],
                      self.opt['fusion']['dim_h'])
            for i in range(self.opt['attention']['nb_glimpses'])])
        self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                         self.opt['fusion']['dim_h']
                                         * self.opt['attention']['nb_glimpses'])
        self.linear_classif = nn.Linear(self.opt['attention']['dim_mm'],
                                        self.num_classes)

        self.bilinear = nn.Bilinear(self.opt['attention']['dim_v']
                                    * self.opt['attention']['nb_glimpses'],
                                    self.opt['attention']['dim_q'],
                                    self.opt['attention']['dim_mm'])

    def _fusion_att(self, x_v, x_q):
        x_att = torch.pow(x_q, 2)
        x_att = torch.mul(x_v, x_att)
        return x_att

    def _fusion_classif(self, x_v, x_q):
        x_q = torch.pow(x_q, 2)
        x_q = x_q.view(x_q.shape[0], int(
            x_q.shape[1]/self.opt['attention']['nb_glimpses']), -1)
        x_q = torch.sum(x_q, dim=2)
        x_mm = self.bilinear(x_v, x_q)
        return x_mm

    def _fusion_classif(self, x_v, x_q):
        x_mm = torch.add(x_v, 4, x_q)
        return x_mm
