from gensim.models import Word2Vec, KeyedVectors
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.utils import class_weight
import torch
from gensim.scripts.glove2word2vec import glove2word2vec
import pandas as pd
import numpy as np
import random
import re
import os,sys
import gensim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau as lr_scheduler
from torch.optim.lr_scheduler import StepLR
import num2words
from num2words import num2words
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unknown_tag = "<unknown_tag>"
number_tag = "<number_tag>"
curr_batch_len = 8

my_num_list = []
for i in range(100):
    my_num_list.append(num2words(i))
other_number = ['hundred', 'thousand', 'million', 'billion', 'trillion', 'quadrillion', 'quintillion', 'sextillion', 'septillion', 'octillion', 'nonillion', 'decillion']
for i in other_number:
    my_num_list.append(i)

   
def process_input(filename):

    f=open(os.path.join(sys.path[0],filename),"r")
    lines=f.read().splitlines()
    curr_sentence =[]
    curr_word = []
    get_all_words = []
    for x in lines:
        y = x.split()

        if len(y)==3:
            curr_sentence.append(y)
            curr_word.append(y[1])
        else:
          get_all_words.append(curr_word)
          curr_word = []
    get_all_words.append(curr_word)
        
    f.close()


    return curr_sentence,get_all_words

def process_test_input(filename):
    f=open(os.path.join(sys.path[0],filename),"r")
    lines=f.read().splitlines()
    curr_sentence =[]
    curr_word = []
    get_all_words = []
    for x in lines:
        y = x.split()
        if len(y)==2:
            curr_sentence.append(y)
            curr_word.append(y[1])
        else:
          get_all_words.append(curr_word)
          curr_word = []
        
    get_all_words.append(curr_word)
    f.close()


    return curr_sentence,get_all_words


pre_train,pre_train_words = process_input("train")
pre_dev,pre_dev_words = process_input("dev")
pre_test,pre_test_words = process_test_input("test")


def chk_num_tag(given_string):
    try:
        
        f=filter(str.isalnum,given_string)
        given_string="".join(f)
        
        if given_string.lower() in my_num_list:
            
            return True
        
        elif given_string.isdecimal():
            return True
        
        float(given_string)
        return True
    except ValueError:
            return False
        
curr_indx = 0
all_ner_tags = []
my_vocab = {}
wordidx_dict = {}
tagidx_dict = {}

for curr_line in pre_train:
    my_word = curr_line[1]
    my_tag = curr_line[2]
    if chk_num_tag(my_word):
        my_word = number_tag
    if my_word in my_vocab:
        my_vocab[my_word] += 1
    else:
        my_vocab[my_word] = 1
    if my_tag not in tagidx_dict:
        tagidx_dict[my_tag] = curr_indx
        
        curr_indx += 1
    all_ner_tags.append(tagidx_dict[my_tag])

my_vocab["<unknown_tag>"] = 0


curr_indx = 0
for my_word in my_vocab:
    if my_word not in wordidx_dict:
        wordidx_dict[my_word] = curr_indx
        
        curr_indx += 1


def make_x_y_blstm(pre_data):
    blstm_word_idx = []
    blstm_tag_idx = []
    list_of_sen = []
    list_of_sen.append(pre_data[0])
    for i in range(1,len(pre_data)):
        if pre_data[i][0] != '1':
            list_of_sen.append(pre_data[i])
        else:
            words = np.array([])
            tags = np.array([])
            my_word =''
            for curr_indx in range(len(list_of_sen)):
                my_word = list_of_sen[curr_indx][1]
                if my_word not in wordidx_dict:
         
                    if chk_num_tag(list_of_sen[curr_indx][1]):
                        my_word  = number_tag

                    else:
                        my_word  = unknown_tag
       
                words= np.append(words,wordidx_dict[my_word])
                tags= np.append(tags,tagidx_dict[list_of_sen[curr_indx][2]])            
            
            blstm_word_idx.append(words)
            blstm_tag_idx.append(tags)

            list_of_sen = []
            list_of_sen.append(pre_data[i])
    
    
    blstm_word_idx.append(np.array([257.]))
    blstm_tag_idx.append(np.array([1]))

    return np.array(blstm_word_idx), np.array(blstm_tag_idx)

train_blstm_x, train_blstm_y = make_x_y_blstm(pre_train)
dev_blstm_x, dev_blstm_y = make_x_y_blstm(pre_dev)

ip_dim = 20178
op_dim = 9
em_dim = 100
hidden_dim = 256
fc_dim = 128



class task1_simple_BLSTM(torch.nn.Module):
    def __init__(self, ip_dim, em_dim, hidden_dim, fc_dim, op_dim):
        super(task1_simple_BLSTM, self).__init__()
        self.embed_layer = torch.nn.Embedding(num_embeddings = ip_dim, embedding_dim = em_dim)
        self.blstm_layer = torch.nn.LSTM(input_size = em_dim, hidden_size = hidden_dim, num_layers = 1, bidirectional = True, batch_first = True, dropout = 0.33)
        self.linear_layer = torch.nn.Linear(hidden_dim*2, fc_dim)
        self.elu_layer = torch.nn.ELU()
        self.classifier_layer = torch.nn.Linear(fc_dim, op_dim)
    def forward(self, x,x_len):
        embed_layer_out = self.embed_layer(x)
        blstm_packed = pack_padded_sequence(embed_layer_out,x_len,batch_first=True,enforce_sorted=True)
        blstm_layer_out, _ = self.blstm_layer(blstm_packed)
        blstm_layer_out, _ = pad_packed_sequence(blstm_layer_out, batch_first=True)
        classifier_layer_out = self.elu_layer(self.linear_layer(blstm_layer_out))
        target_output = self.classifier_layer(classifier_layer_out)
        return target_output

    
    def wt_initalize(self):
        for name, param in self.named_parameters():
            torch.nn.init.normal_(param.data, mean=0, std=0.1)





def make_output_file(x, y, pre_data,loaded_model,op_filename, test_file = False):
   
    loaded_model.eval()
    line_num = 0
    me = 0
    
    with torch.no_grad():
        with open(op_filename, "w") as fp:
            for i in range(len(x)):
              curr_indx = 1
              x_len = np.array([len(x[i])])
              ip = torch.LongTensor(x[i]).to(device)
              ip = torch.unsqueeze(ip, 0)
              op = loaded_model(ip,x_len)
              op = op.view(-1, op.shape[-1])
              _, pred = torch.max(op, 1)

              if not test_file:
                targ_ner = y[i]
                for j in range(len(targ_ner)):
                    
                    pred_tag = int(pred[j])
                    targ_tag = int(targ_ner[j])
                    
                    z = pre_data[line_num][1]
                    
                    for key,value in tagidx_dict.items():
                      if value == targ_tag:
                        gold = key
                      if value == pred_tag:
                        my_word = key
                    
                    my_sen_pred = str(curr_indx)+' '+z+' '+my_word+'\n'
                    fp.write(my_sen_pred)
                    curr_indx += 1
                    line_num += 1
                fp.write('\n')
              else:
                for j in range(len(pred)):
                  pred_tag = int(pred[j])
                  z = pre_data[line_num][1]
                  for key,value in tagidx_dict.items():
                      if value == pred_tag:
                        my_word = key
                  my_sen_pred = str(curr_indx)+' '+z+' '+my_word+'\n'
                  fp.write(my_sen_pred)
                  curr_indx += 1
                  line_num += 1
                fp.write('\n')




load_model = task1_simple_BLSTM(ip_dim, em_dim, hidden_dim, fc_dim, op_dim).to(device)
load_model.load_state_dict(torch.load(os.path.join(sys.path[0],'blstm1.pt'),map_location='cpu') )
make_output_file(dev_blstm_x, dev_blstm_y, pre_dev, load_model, os.path.join(sys.path[0],'dev1.out'))

def make_test_x_y_blstm(pre_data):
    blstm_word_idx = []
    
    list_of_sen = []
    list_of_sen.append(pre_data[0])
    for i in range(1,len(pre_data)):
        if pre_data[i][0] != '1':
            list_of_sen.append(pre_data[i])
        else:
            words = np.array([])
            
            my_word =''
            for curr_indx in range(len(list_of_sen)):
                my_word = list_of_sen[curr_indx][1]
                if my_word not in wordidx_dict:
         
                    if chk_num_tag(list_of_sen[curr_indx][1]):
                        my_word  = number_tag

                    else:
                        my_word  = unknown_tag
       
                words= np.append(words,wordidx_dict[my_word])
                
            
            blstm_word_idx.append(words)
            

            list_of_sen = []
            list_of_sen.append(pre_data[i])
    
    
    blstm_word_idx.append(np.array([257.]))
   

    return np.array(blstm_word_idx)



test_blstm_x = make_test_x_y_blstm(pre_test)
del load_model
load_model = task1_simple_BLSTM(ip_dim, em_dim, hidden_dim, fc_dim, op_dim).to(device)
load_model.load_state_dict(torch.load(os.path.join(sys.path[0],'blstm1.pt'),map_location='cpu'))
make_output_file(test_blstm_x, [], pre_test, load_model, os.path.join(sys.path[0],'test1.out'),True)


all_words = pre_train_words + pre_dev_words + pre_test_words
glove_w2v_file = os.path.join(sys.path[0],'glove.txt.word2vec.out')
glove2word2vec(os.path.join(sys.path[0],'glove.6B.100d.gz'), glove_w2v_file)
glove_vec = KeyedVectors.load_word2vec_format(glove_w2v_file)


all_wordidx_dict = {}
curr_indx = 0

for t2_curr_word in all_words:
    for my_word in t2_curr_word:
        if my_word not in all_wordidx_dict:
            all_wordidx_dict[my_word] = curr_indx
            curr_indx += 1
    all_wordidx_dict[number_tag] = curr_indx


embed_weight_mat = np.zeros((len(all_wordidx_dict), 100), dtype = float)
glove_embed_dict = {}


def glove_embed_weight(embed_weight_mat, model):
    global all_wordidx_dict, glove_embed_dict
    for my_word, curr_indx in all_wordidx_dict.items():
        my_word = my_word.lower()
        
        if my_word in glove_embed_dict:
            embed_weight_mat[curr_indx] = glove_embed_dict[my_word]
        else:
          if my_word in model.vocab:
            embed_weight_mat[curr_indx] = model[my_word]
          else:
            rand_embed = np.random.normal(scale = 0.6, size = (100,))
            embed_weight_mat[curr_indx] = rand_embed
            glove_embed_dict[my_word] = rand_embed

            
    
    return embed_weight_mat


embed_weight_mat = glove_embed_weight(embed_weight_mat, glove_vec)

def glove_make_x_y_blstm(pre_data,test = False):
    blstm_word_idx = []
    blstm_tag_idx = []
    list_of_sen = []
    list_of_sen.append(pre_data[0])
    for i in range(1,len(pre_data)):
        if pre_data[i][0] != '1':
            list_of_sen.append(pre_data[i])
        else:
            words = np.array([])
            tags = np.array([])
            my_word =''
            for curr_indx in range(len(list_of_sen)):
                my_word = list_of_sen[curr_indx][1]
                if chk_num_tag(my_word):

                  my_word  = number_tag              
       
                words= np.append(words,all_wordidx_dict[my_word])
                if not test:
                    tags= np.append(tags,tagidx_dict[list_of_sen[curr_indx][2]])            
            
            blstm_word_idx.append(words)
            if not test:
                blstm_tag_idx.append(tags)

            list_of_sen = []
            list_of_sen.append(pre_data[i])
    
    
    blstm_word_idx.append(np.array([257.]))
    if not test:
        blstm_tag_idx.append(np.array([1]))

    return np.array(blstm_word_idx), np.array(blstm_tag_idx)

glove_train_blstm_x, glove_train_blstm_y = glove_make_x_y_blstm(pre_train)
glove_dev_blstm_x, glove_dev_blstm_y = glove_make_x_y_blstm(pre_dev)
glove_test_blstm_x,_ = glove_make_x_y_blstm(pre_test,True)

ip_dim = len(all_wordidx_dict)
em_dim = 100
hidden_dim = 256
fc_dim = 128
op_dim = len(tagidx_dict)
tag_weightage = class_weight.compute_class_weight('balanced', np.unique(all_ner_tags), all_ner_tags)

def find_embed_layer(ip_dim, em_dim, embed_weight_mat):
    embed_weight_mat = torch.FloatTensor(embed_weight_mat).to(device)
    embed_layer = torch.nn.Embedding(num_embeddings = ip_dim, embedding_dim = em_dim)
    embed_layer.load_state_dict({'weight': embed_weight_mat})
    return embed_layer


class task2_glove_BLSTM(torch.nn.Module):
    def __init__(self, ip_dim, em_dim, hidden_dim, fc_dim, op_dim, embed_weight_mat):
        super(task2_glove_BLSTM, self).__init__()
        self.embed_layer = find_embed_layer(ip_dim, em_dim, embed_weight_mat)
        self.blstm_layer = torch.nn.LSTM(input_size = em_dim, hidden_size = hidden_dim, num_layers = 1, bidirectional = True, batch_first = True, dropout = 0.33)
        self.linear_layer = torch.nn.Linear(hidden_dim * 2, fc_dim)
        self.elu_layer = torch.nn.ELU()
        self.classifier_layer = torch.nn.Linear(fc_dim, op_dim)
    
    def forward(self, x,x_len):
        emb = self.embed_layer(x)
        blstm_packed = pack_padded_sequence(emb, x_len, batch_first = True, enforce_sorted = True)
        blstm_layer_out, _ = self.blstm_layer(blstm_packed)
        blstm_layer_out, _ = pad_packed_sequence(blstm_layer_out, batch_first = True)
        classifier_layer_out = self.elu_layer(self.linear_layer(blstm_layer_out))
        target_output = self.classifier_layer(classifier_layer_out)
        return target_output
    
load_model = task2_glove_BLSTM(ip_dim, em_dim, hidden_dim, fc_dim, op_dim,embed_weight_mat).to(device)
load_model.load_state_dict(torch.load(os.path.join(sys.path[0],'blstm2.pt'),map_location='cpu' ))
make_output_file(glove_dev_blstm_x, glove_dev_blstm_y, pre_dev, load_model, os.path.join(sys.path[0],'dev2.out'))

del load_model

load_model = task2_glove_BLSTM(ip_dim, em_dim, hidden_dim, fc_dim, op_dim,embed_weight_mat).to(device)
load_model.load_state_dict(torch.load(os.path.join(sys.path[0],'blstm2.pt'),map_location='cpu' ))
make_output_file(glove_test_blstm_x, [], pre_test, load_model, os.path.join(sys.path[0],'test2.out'),True)

