from transformers import BertTokenizer, BertModel, T5Tokenizer, T5Model
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

class CombinedModel(nn.Module):
    def __init__(self, bert_model_name, t5_model_name, num_classes,device):
        super(CombinedModel, self).__init__()

        # self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        # self.bert_model = BertModel.from_pretrained(bert_model_name).to(device)
        # self.t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')

        self.bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
        self.t5_model = AutoModelForSequenceClassification.from_pretrained('t5-base').to(device)

        # 去掉BERT和T5的分类器，用来得到以藏层
        self.bert_model.classifier = nn.Identity()
        self.t5_model.classification_head.out_proj=nn.Identity()

        self.device = device

        # 创建一个额外的全连接层分类
        self.global_conv = nn.Conv2d(2,1,15,7,0)
        self.global_classifier = nn.Sequential(nn.Linear(1536, 384),
                                                nn.Linear(384, num_classes)
                                               ).to(device)

        self.loss = nn.CrossEntropyLoss()
        self.step = 0
        self.acc = 0
        self.added_acc = 0

        self.stop = 20

    def forward(self, batch_bert,batch_t5):
        # 使用BERT模型进行特征提取
        
        input_ids = batch_bert['input_ids'].to(self.device)
        attention_mask = batch_bert['attention_mask'].to(self.device)
        labels = batch_bert['label'].to(self.device)
        bert_outputs = self.bert_model(input_ids, attention_mask=attention_mask)
        # bert_loss = bert_outputs.loss
        # bert_last_hidden_state = bert_outputs.last_hidden_state
        # t5_input_ids = self.t5_tokenizer.encode(text, return_tensors="pt")

        # 使用T5模型进行特征提取
        input_ids = batch_t5['input_ids'].to(self.device)
        attention_mask = batch_t5['attention_mask'].to(self.device)
        labels = batch_t5['label'].to(self.device)
        t5_outputs = self.t5_model(input_ids, attention_mask=attention_mask)
        # t5_loss = t5_outputs.loss
        # t5_last_hidden_state = t5_outputs.encoder_last_hidden_state
        

        #JIANET 核心部分在下面
        ###########################################################
        # 将BERT和T5的特征层拼接在一起
        
        bert_vector = bert_outputs.logits.unsqueeze(dim=2)  # b,768,1
        t5_vector = t5_outputs.logits.unsqueeze(dim=1)      # b,1,768

        interaction_map = bert_vector @ t5_vector           #b,768,768

        bert_vector_T = bert_outputs.logits.unsqueeze(dim=1)  # b, 1, 768
        t5_vector_T = t5_outputs.logits.unsqueeze(dim=2)      # b, 768, 1

        bert2t5 = bert_vector_T @ interaction_map           #b,1,768
        t52bert = interaction_map @ t5_vector_T             #b,768,1


        concat_last_hidden_state = torch.cat([bert2t5.squeeze(), t52bert.squeeze()], dim=-1)

        ###########################################################

        # 使用全连接层进行分类
        logits = self.global_classifier(concat_last_hidden_state)

        # direct_logits = t5_outputs.logits + bert_outputs.logits
        direct_logits = 0

        self_loss = self.loss(logits,labels)

        final_loss = self_loss

        self.step +=1

        self.acc += (logits.argmax(dim=1)==labels).sum()/(32)

        #阶段性的输出测试过程参数，可以去掉/
        if(self.step%self.stop==0):
            # print("concat_acc = "+str(self.acc/self.stop) + "added_acc = " + str(self.added_acc/self.stop) + "loss " + str(final_loss))
            
            self.added_acc=0
            self.acc = 0
            
        return logits,final_loss , direct_logits

