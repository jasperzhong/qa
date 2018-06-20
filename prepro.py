import json

import jieba.posseg as pseg

from utils import *


def prepro(config):
    '''
    save {'questions':[[(词,词性),(词,词性)],[]...,[]], 'contexts':[], 'answers':[], 'offsets',[[start,end],...[]]}
    '''
    train_data = load_dataset(config.resource.train_data_dir)
    dev_data = load_dataset(config.resource.dev_data_dir)

    print('process train data...')
    data = process(config, train_data)
    with open(config.resource.train_preprocess_dir, 'w', encoding='utf-8') as f:
        f.write(data)
    print('process dev data...')
    data = process(config, dev_data)
    with open(config.resource.dev_preprocess_dir, 'w', encoding='utf-8') as f:
        f.write(data)

    
    
def process(config, data):
    qas = data["questions"]
    ans = data["answers"]
    cts = data["contexts"]

    qas_pos = []
    ans_pos = []
    cts_pos = []
    offsets = [] #start end
    for qa, an, ct in zip(qas, ans, cts):
        words = pseg.cut(qa)
        qas_pos.append([(word,flag) for word, flag in words])
        
        start = an[0]['answer_start']
        end = an[0]['answer_start']+len(an[0]['text'])
        

        words = pseg.cut(an[0]['text'])
        ans_pos.append([(word,flag) for word, flag in words])
        
        ct1 = ct[:start]
        ct2 = ct[end:]
        words1 = pseg.cut(ct1)
        words = pseg.cut(ct[start:end])
        words2 = pseg.cut(ct2)
        tmp = [(word,flag) for word, flag in words1]
        start = len(tmp)
        tmp.extend([(word,flag) for word, flag in words])
        end = len(tmp) - 1
        tmp.extend([(word,flag) for word, flag in words2])
        offsets.append((start, end))
        cts_pos.append(tmp)
    

    data = {'questions':qas_pos, 
                  'answers': ans_pos,
                  'contexts': cts_pos,
                  'offsets': offsets}

    data = json.dumps(data, ensure_ascii=False)

    return data
    