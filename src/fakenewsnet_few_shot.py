import glob, json, random, re, nltk
from openprompt.data_utils import InputExample, FewShotSampler
from openprompt.prompts import MixedTemplate
from openprompt.plms import load_plm
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import SoftVerbalizer, ManualVerbalizer
import torch
from transformers import  AdamW, get_linear_schedule_with_warmup



def clean(txt):
    return txt.encode("ascii", "ignore").decode('utf-8').replace("\n", "")

def test_set(all, train, dev, few_shot=True):
    test = []
    if few_shot:
        used_id = []
        for idx in range(len(train)):
            used_id.append(train[idx].guid)
            used_id.append(dev[idx].guid)

        for idx in range(len(all)):
            if all[idx].guid not in used_id:
                test.append(all[idx])
            else:
                continue
    return test




# fake_file_poli = glob.glob("../FakeNewsNet-master/code/fakenewsnet_dataset/politifact/fake/*/*.json")
fake_file_goss = glob.glob("../FakeNewsNet-master/code/fakenewsnet_dataset/gossipcop/fake/*/*.json")
# fake_files = fake_file_poli + fake_file_goss
fake_files = fake_file_goss

# real_file_poli = glob.glob("../FakeNewsNet-master/code/fakenewsnet_dataset/politifact/real/*/*.json")
real_file_goss = glob.glob("../FakeNewsNet-master/code/fakenewsnet_dataset/gossipcop/real/*/*.json")
# rea_files = real_file_poli + real_file_goss
real_files = real_file_goss

# print(len(fake_files)) #5159
# print(len(rea_files)) #11361

data = []

for file in fake_files:
    with open(file, "r") as inf:
        fake_data = json.load(inf)
        fake_data['label'] = 1
        data.append(fake_data)

for file in real_files:
    with open(file, "r") as inf2:
        real_data = json.load(inf2)
        real_data['label'] = 0
        data.append(real_data)

# print(data[0]['images'])
# print(len(data)) #16520
random.shuffle(data)

dataset = []
for idx, d in enumerate(data):

    input_example = InputExample(text_a=clean(d['title']), text_b=clean(d['text']), label=int(d['label']), guid=str(idx))
    dataset.append(input_example)

sampler  = FewShotSampler(num_examples_per_label=1, num_examples_per_label_dev=1, also_sample_dev=True)
train, dev = sampler(dataset)

# print(len(train)) #2
# print(len(dev)) #2

test = test_set(dataset, train, dev)

plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")

mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer,
                            text='{"soft"}Here is a piece of news with {"mask"} information.{"soft"} {"placeholder":"text_a"} {"placeholder":"text_b"}')
# wrapped_example = mytemplate.wrap_one_example(train[0])
# print(wrapped_example)

train_dataloader = PromptDataLoader(dataset=train, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
    batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")


# for example the verbalizer contains multiple label words in each class
myverbalizer = ManualVerbalizer(tokenizer=tokenizer, classes=['fake', 'real'],
         label_words={
         "fake": ["false", "fake", "unreal", "misleading",
                     "artificial", "bogus", "virtual", "incorrect",
                     "wrong", "fault"],
        "real": ["true", "real", "actual", "substantial",
                     "authentic", "genuine", 'factual','correct',
                     "fact", "truth"]
         })
# or without label words
# myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=2)

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

loss_func = torch.nn.CrossEntropyLoss()

no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters

# optimizer_grouped_parameters2 = [
#     {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
#     {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
# ]


optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
# optimizer2 = AdamW(optimizer_grouped_parameters2)


for epoch in range(5):
    tot_loss = 0
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        tot_loss += loss.item()
        optimizer1.step()
        optimizer1.zero_grad()
        # optimizer2.step()
        # optimizer2.zero_grad()
        print(tot_loss/(step+1))

# ## evaluate

# %%
validation_dataloader = PromptDataLoader(dataset=dev, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
    batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

prompt_model.eval()

allpreds = []
alllabels = []
for step, inputs in enumerate(validation_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print("validation:",acc)


test_dataloader = PromptDataLoader(dataset=test, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
    batch_size=4, shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")
allpreds = []
alllabels = []
for step, inputs in enumerate(test_dataloader):
    if use_cuda:
        inputs = inputs.cuda()
    logits = prompt_model(inputs)
    labels = inputs['label']
    alllabels.extend(labels.cpu().tolist())
    allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
print("test:", acc)