import os



class ResourceConfig(object):
    cur_dir = os.path.abspath(__file__)

    d = os.path.dirname
    base_dir = d(cur_dir)
    data_dir = os.path.join(base_dir, "data")

    train_data_dir = os.path.join(data_dir, "train-v1.1.json")
    dev_data_dir = os.path.join(data_dir, "dev-v1.1.json")
    test_data_dir = os.path.join(data_dir, "test-v1.1.json")

    train_preprocess_dir = os.path.join(data_dir, "train_preprocess.json")
    dev_preprocess_dir = os.path.join(data_dir, "dev_preprocess.json")

    model_save_dir = os.path.join(data_dir, "model", "naive.mdl")

    embedding_dir = os.path.join(data_dir, "word2vec.txt")
    #stanfordcorenlp_path = "D:/Tools/stanford_corenlp"


class ModelConfig(object):
    vocab_size = 0  #得建立完词汇表后才能确定
    embedding_dim = 300
    num_features = 3 
    hidden_size = 128 #lstm hidden size 
    passage_layers = 3 #same with drqa
    dropout_rate = 0.4 #same
    question_layers = 3 #same with drqa


class TrainingConfig(object):
    OPT = ['sgd', 'adam', 'admax']
    batch_size = 32 #default 
    epoch = 40
    lr = 0.1
    momentum = 0.9
    l2 = 1e-5 
    weight_decay = 0.999
    max_len = 15

    
class Config(object):
    CMD = ['train', 'evaluate', 'prepro', 'interactive']

    training = TrainingConfig()
    resource = ResourceConfig()
    model = ModelConfig()
    



    
