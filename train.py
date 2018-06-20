import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import Reader
from utils import *


class Trainer(object):
    def __init__(self, config):
        self.config = config
        self.dictionary = Dictionary()
        self.pos_dict = PosDict()
        self.total_words = 0
        self.model = None

    def start(self):
        print('building dictionary...')
        self.build_dict()
        self.config.model.vocab_size = self.dictionary.n_words

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.model = torch.load(self.config.resource.model_save_dir)
            print('load previous model...')
        except FileNotFoundError:
            print("build from scratch...")
            self.model = Reader(self.config).to(device)

        print('init embedding...')
        #self.model.load_embeddings(self.dictionary.word_to_ix, self.config.resource.embedding_dir)
        
        print('vectorize...')
        train_X1, train_X1_f, train_X2, train_Y = self.vectorize(self.train_data)
        dev_X1, dev_X1_f, dev_X2, dev_Y = self.vectorize(self.dev_data)

        loss_func = nn.NLLLoss()
        optimizer = optim.SGD(self.model.parameters(), 
            lr=self.config.training.lr, 
            momentum=self.config.training.momentum)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        print('start training!')
        for epoch in range(self.config.training.epoch):
            total_loss = 0
            cnt = 0
            
            for x1, x1_f, x2, y in zip(train_X1, train_X1_f, train_X2, train_Y):
                self.model.zero_grad()
                x1 = torch.LongTensor([x1]).to(device)
                x1_f = torch.Tensor([x1_f]).to(device)
                x2 = torch.LongTensor([x2]).to(device)
                
                start, end = self.model(x1, x1_f, x2)
                loss1 = loss_func(start, torch.LongTensor([y[0]]).to(device))
                loss2 = loss_func(end, torch.LongTensor([y[1]]).to(device))
                loss = loss1 + loss2 
                total_loss += loss.item()
                loss.backward()
                scheduler.step()
                cnt += 1
            print("epoch %d: train loss %.2f" % (epoch, total_loss/cnt))
        
            with torch.no_grad():
                cnt = 0
                f1 = 0
                for x1, x1_f, x2, y in zip(dev_X1, dev_X1_f, dev_X2, dev_Y):
                    x1 = torch.LongTensor([x1]).to(device)
                    x1_f = torch.Tensor([x1_f]).to(device)
                    x2 = torch.LongTensor([x2]).to(device)
                    
                    score_s, score_e = self.model(x1, x1_f, x2)
                    pred_s, pred_e, pred_score = self.decode(score_s.cpu(), score_e.cpu(), top_n=1, max_len=15)
                    prediction = ''.join([word  for word,flag in self.dev_data['contexts'][cnt][pred_s[0][0]:pred_e[0][0]]])
                    f1 += f1_score(prediction, self.dev_data['answers'][cnt][0])
                    cnt += 1
                print("epoch %d: dev f1 %.2f" % (epoch, f1*100.0/cnt))
                    
                              

    def decode(self, score_s, score_e, top_n=1, max_len=None):
        pred_s = []
        pred_e = []
        pred_score = []
        max_len = max_len or score_s.size(1)
        for i in range(score_s.size(0)):
            # Outer product of scores to get full p_s * p_e matrix
            scores = torch.ger(score_s[i], score_e[i])

            # Zero out negative length and over-length span scores
            scores.triu_().tril_(max_len - 1)

            # Take argmax or top n
            scores = scores.numpy()
            scores_flat = scores.flatten()
            if top_n == 1:
                idx_sort = [np.argmax(scores_flat)]
            elif len(scores_flat) < top_n:
                idx_sort = np.argsort(-scores_flat)
            else:
                idx = np.argpartition(-scores_flat, top_n)[0:top_n]
                idx_sort = idx[np.argsort(-scores_flat[idx])]
            s_idx, e_idx = np.unravel_index(idx_sort, scores.shape)
            pred_s.append(s_idx)
            pred_e.append(e_idx)
            pred_score.append(scores_flat[idx_sort])
        return pred_s, pred_e, pred_score

    def build_dict(self):
        with open(self.config.resource.train_preprocess_dir, encoding='utf-8') as f:
            self.train_data = json.load(f)
        
        with open(self.config.resource.dev_preprocess_dir, encoding='utf-8') as f:
            self.dev_data = json.load(f)

        for qa in self.train_data["questions"]:
            for q in qa:
                self.dictionary.addWord(q[0])
                self.pos_dict.addPos(q[1])
                self.total_words += 1
        for an in self.train_data["answers"]:
            for q in an:
                self.dictionary.addWord(q[0])
                self.pos_dict.addPos(q[1])
                self.total_words += 1
        for ct in self.train_data["contexts"]:
            for q in ct:
                self.dictionary.addWord(q[0])
                self.pos_dict.addPos(q[1])
                self.total_words += 1
        for qa in self.dev_data["questions"]:
            for q in qa:
                self.dictionary.addWord(q[0])
                self.pos_dict.addPos(q[1])
                self.total_words += 1
        for an in self.dev_data["answers"]:
            for q in an:
                self.dictionary.addWord(q[0])
                self.pos_dict.addPos(q[1])
                self.total_words += 1
        for ct in self.dev_data["contexts"]:
            for q in ct:
                self.dictionary.addWord(q[0])
                self.pos_dict.addPos(q[1])
                self.total_words += 1

    def vectorize(self, data):
        '''
        transform input to tensor

        each token in passage: idx + exact_match + pos + tf
        each token in question: idx 
        X [x1, x1_f, x2]
        y [start, end] 
        '''
        X1 = []  #idx int 1位
        X1_f = []  #exact_match bool 1位 + pos_idx int 1位 + tf float 1位
        X2 = [] #idx int 1位
        Y = [] 
        for qa, ct, off in zip(data['questions'], data['contexts'], data['offsets']):
            qa_word_set = set([q[0] for q in qa])
            x1 = []
            x1_f = []

            for c in ct:
                x1.append(self.dictionary.word_to_ix[c[0]])
                f = [0, 0, 0]
                if c[0] in qa_word_set:
                    f[0] = 1
                f[1] = self.pos_dict.pos_to_ix[c[1]]
                f[2] = float(self.dictionary.word_to_cnt[c[0]]) / self.total_words
                x1_f.append(f)
                    
            X1.append(x1)
            X1_f.append(x1_f)

            x2 = []
            for q in qa:
                x2.append(self.dictionary.word_to_ix[q[0]])
            X2.append(x2)

            Y.append(off)
        
        return (
            X1, X1_f, X2, Y
        )
