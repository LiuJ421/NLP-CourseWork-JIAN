from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import  AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset


from tqdm.auto import tqdm
from utils.ensemblemodels import CombinedModel
import torch.nn as nn

# 检查是否有可用的CUDA设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载SST-2数据集
dataset = load_dataset('glue', 'sst2')


# 加载分词器和模型
tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
tokenizer_t5 = AutoTokenizer.from_pretrained('t5-base')
model = CombinedModel('bert-base-uncased', 't5-base', num_classes=2,device=device)
model.load_state_dict(torch.load('weights/ensembleJIAN9415.pth'),strict=False)



# 定义最大序列长度
max_length = 128

# 将数据集转换为模型可接受的形式
def tokenize_bert(batch):
    return tokenizer_bert(batch['sentence'], padding='max_length', truncation=True, max_length=max_length)

def tokenize_t5(batch):
    return tokenizer_t5(batch['sentence'], padding='max_length', truncation=True, max_length=max_length)

dataset_bert = dataset.map(tokenize_bert, batched=True)
dataset_t5 = dataset.map(tokenize_t5, batched=True)

dataset_bert.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
dataset_t5.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# 初始化数据加载器
data_loader_bert = DataLoader(dataset_bert['train'], batch_size=32)
data_loader_t5 = DataLoader(dataset_t5['train'], batch_size=32)


# 初始化优化器和学习率调度器

num_epochs = 5
num_training_steps = num_epochs * len(data_loader_bert)

##只训练选中的曾
# for k,v in model.named_parameters():
#     v.requires_grad = False
#     # if 't5_model.classification_head'  in k:
#     #     v.requires_grad = True
#     # if 'bert_model.classifier' in k :
#     #     v.requires_grad = True

#     if 'global_classifier' in k :
#         # print(k)
#         v.requires_grad = True
#         # print("tes")

# model.bert_model.classifier.requires_grad_=False
# model.t5_model.classification_head.out_proj.require_grad_ = False
#model.bert_model.classifier.parameters()    model.t5_model.classification_head.out_proj.parameters()

optimizer = AdamW(model.parameters(), lr=5e-5)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# evaluate 函数
evaluator_bert = DataLoader(dataset_bert['validation'], batch_size=16)
evaluator_t5 = DataLoader(dataset_t5['validation'], batch_size=16)

def eval():
    model.eval()

    correct_concat = 0
    correct_add = 0
    total = 0

    with torch.no_grad():
        for batch_bert,batch_t5 in zip(evaluator_bert,evaluator_t5):
            
            labels = batch_bert['label'].to(device)
            outputs = model(batch_bert,batch_t5)
            #logits that concat
            logits = outputs[0]
            predictions = torch.argmax(logits, dim=-1)
            correct_concat += (predictions == labels).sum().item()

            total += len(labels)

    accuracy_concat = correct_concat / total

    print(f"Accuracy concat: {accuracy_concat * 100:.2f}%")
    model.train()
    return accuracy_concat

eval()

max_acc=0

for epoch in range(num_epochs):
    model.train()
    progress_bar_bert = tqdm(data_loader_bert, desc=f"Epoch {epoch+1}/{num_epochs}")
    progress_bar_t5 = tqdm(data_loader_t5, desc=f"Epoch {epoch+1}/{num_epochs}")

    step = 0

    for batch_bert, batch_t5 in zip(progress_bar_bert, progress_bar_t5):
        step +=1
        outputs = model(batch_bert,batch_t5)
        loss = outputs[1]

        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    torch.save(model.state_dict(), 'weights/ensemble.pth')

    model.load_state_dict(torch.load('weights/ensemble.pth'))

    acc= eval()
