import torch 
import torch.nn as nn
from torch.nn import init

import layers
import tqdm

class Reader(nn.Module):
    def __init__(self, config):
        super(Reader, self).__init__()
        self.config = config.model

        #word embedding
        self.embedding = nn.Embedding(self.config.vocab_size, 
                                    self.config.embedding_dim)

        #attention weighted question
        self.qemb_match = layers.SeqAttnMatch(self.config.embedding_dim)
        self.qemb_match.linear.data.normal_(0, 1)

        self.passage_input_size = self.config.embedding_dim + self.config.num_features + self.config.embedding_dim
        self.question_input_size = self.config.embedding_dim
        self.passage_encoder = layers.StackedBiLSTM(
            input_size=self.passage_input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.passage_layers,
            dropout_rate=self.config.dropout_rate
        )
        self.passage_encoder.rnns.data.normal_(0, 1)

        self.question_encoder = layers.StackedBiLSTM(
            input_size=self.question_input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.question_layers,
            dropout_rate=self.config.dropout_rate
        )
        self.question_encoder.rnns.data.normal_(0, 1)
        
        #question merging
        self.self_attn = layers.LinearSeqAttn(self.config.hidden_size)
        self.self_attn.linear.data.normal_(0, 1)

        #span start/end
        self.start_attn = layers.BilinearSeqAttn(
            self.config.hidden_size,
            self.config.hidden_size
        )
        self.start_attn.linear.data.normal_(0, 1)

        self.end_attn = layers.BilinearSeqAttn(
            self.config.hidden_size,
            self.config.hidden_size
        )
        self.end_attn.linear.data.normal_(0, 1)
    
    def forward(self, x1, x1_f, x2):
        '''
        inputs:
        x1: document word indices  [batch * len_d]
        x1_f: document word featers indices [batch * len_d * nfeat]
        x2: question word indices [batch * len_q]

        outputs:
        start_scores, end_scores
        '''

        '''
        process:
        1. embedding (nn.embedding)
        2. attention weighted question (layers.SeqAttnMatch)
        ---------input is ready--------
        3. encode document -> {p1,p2,...,pn}  (layers.StackedBiLSTM)
        4. encode question + merge hidden -> q  (layers.StackedBiLSTM)
        5. predict start and end (layers.BilinearSeqAttn)
        '''
        x1_emb = self.embedding(x1)
        x2_emb = self.embedding(x2)

        drnn_input = [x1_emb]
        x2_weighted_emb = self.qemb_match(x1_emb, x2_emb)
        drnn_input.append(x2_weighted_emb)

        drnn_input.append(x1_f)

        drnn_input = torch.cat(drnn_input, 2)

        passage_hiddens = self.passage_encoder(drnn_input)

        question_hiddens = self.question_encoder(x2_emb)

        question_merge_weights = self.self_attn(question_hiddens)
        question_hidden = question_merge_weights.unsqueeze(1).bmm(question_hiddens).squeeze(1)

        start_scores = self.start_attn(passage_hiddens, question_hidden)
        end_scores = self.end_attn(passage_hiddens, question_hidden)

        return start_scores, end_scores

    def load_embeddings(self, word_dict, embedding_file):
        '''
        use pre-trained embeddings  word2vec
        '''
        embedding = self.embedding.weight.data

        with open(embedding_file, encoding='utf-8') as f:
            for line in f:
                parsed = line.rstrip().split(' ')
                assert(len(parsed) == embedding.size(1) + 1)
                w = parsed[0]
                if w in word_dict:
                    vec = torch.Tensor([float(i) for i in parsed[1:]])
                    embedding[word_dict[w]].copy_(vec)