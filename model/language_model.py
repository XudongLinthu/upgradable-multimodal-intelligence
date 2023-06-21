import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1, keepdim=True) / torch.clamp(input_mask_expanded.sum(1, keepdim=True), min=1e-9)

class Bert(nn.Module):
    """ Finetuned DistilBERT module """

    def __init__(self,args=None):
        super(Bert, self).__init__()
        if args.lm == 'all-mpnet-base-v2':
            from transformers import AutoTokenizer, AutoModel
            self.bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.bert = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            self.cls_token = self.bert_tokenizer.cls_token_id
            self.sep_token = self.bert_tokenizer.sep_token_id
            self.lm = 'all-mpnet-base-v2' 
        else:    
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.cls_token = self.bert_tokenizer.cls_token_id
            self.sep_token = self.bert_tokenizer.sep_token_id
            self.lm = 'all-mpnet-base-v2' 

    def forward(self, tokens):
        attention_mask = (tokens > 0).float()
        embds = self.bert(tokens, attention_mask=attention_mask)
        embds = mean_pooling(embds, attention_mask)
        return embds


class Sentence_Maxpool(nn.Module):
    """ Utilitary for the answer module """

    def __init__(self, word_dimension, output_dim, relu=True):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)
        self.out_dim = output_dim
        self.relu = relu

    def forward(self, x_in):
        x = self.fc(x_in)
        x = torch.max(x, dim=1)[0]
        if self.relu:
            x = F.relu(x)
        return x


class AModel(nn.Module):
    """
    Answer embedding module
    """

    def __init__(self, out_dim=512, sentence_dim=2048, args=None):
        super(AModel, self).__init__()
        self.bert = Bert(args)
        self.linear_text = nn.Linear(args.word_dim if not args == None else 768, out_dim) if not (args.baseline == 'to' or args.baseline == 'pt') else nn.Identity()
        self.args = args

    def forward(self, answer):
        if len(answer.shape) == 3:
            bs, nans, lans = answer.shape
            answer = answer.view(bs * nans, lans)
            answer = self.bert(answer)
            answer = answer[:, 0, :]
            answer = answer.view(bs, nans, self.args.word_dim if not self.args == None else 768)
        else:
            answer = self.bert(answer)
            answer = answer[:, 0, :]
        answer = self.linear_text(answer)
        
        return answer
