import json
import random
import re
import string
from collections import Counter

import jieba


class Dictionary(object):
    def __init__(self):
        self.word_to_ix = {}
        self.word_to_cnt = {}
        self.ix_to_word = {}
        self.n_words = 0
    
    def addSentence(self, sentence):
        for word in jieba.cut(sentence):
            self.addWord(word.strip())
    
    def addWord(self, word):
        if word not in self.word_to_ix:
            self.word_to_ix[word] = self.n_words
            self.word_to_cnt[word] = 1
            self.ix_to_word[self.n_words] = word
            self.n_words += 1
        else:
            self.word_to_cnt[word] += 1
    
class PosDict(object):
    def __init__(self):
        self.pos_to_ix = {}
        self.ix_to_pos = {}
        self.n_pos = 0
    
    def addPos(self, pos):
        if pos not in self.pos_to_ix:
            self.pos_to_ix[pos] = self.n_pos
            self.ix_to_pos[self.n_pos] = pos
            self.n_pos += 1


class Batchgen(object):
    '''generate batches
    data is a list
    batch_size is int type
    不过似乎需要按照长度排序才有用.......
    '''
    def __init__(self, data, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data = data
        if shuffle:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.batches:
            yield batch
        raise StopIteration





stop_punc = "+x/=@#$^&*\\?!.,<>;:`-" + "！？——·@#￥……&*+|" + "《》。，、：；%"

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(stop_punc)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_punc(s))


def load_dataset(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)['data']

    output = {'qids': [], 'questions': [], 'answers': [],
              'contexts': [], 'qid2cid': []}
    for article in data:
        for paragraph in article['paragraphs']:
            output['contexts'].append(paragraph['context'])
            for qa in paragraph['qas']:
                if 'answers' in qa:
                    output['answers'].append(qa['answers'])
                else:
                    continue
                output['qids'].append(qa['id'])
                output['questions'].append(qa['question'])
                output['qid2cid'].append(len(output['contexts']) - 1)
                
    return output

def f1_score(prediction, ground_truth):
    """Compute the geometric mean of precision and recall for answer tokens."""
    prediction_tokens = list(normalize_answer(prediction))
    ground_truth_tokens = list(normalize_answer(ground_truth))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

if __name__=='__main__':
    data = load_dataset(r"D:\Program\QA\Question\data\result\test.json")
    print(data['answers'][0])
