import glob, json, random, re, nltk, csv
import numpy as np
from openprompt.data_utils import InputExample, FewShotSampler
from openprompt.prompts import MixedTemplate
from openprompt.plms import load_plm
from openprompt import PromptDataLoader, PromptForClassification
from openprompt.prompts import SoftVerbalizer, ManualVerbalizer
import torch
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import clip
from PIL import Image

def xrange(x):
    return iter(range(x))

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


## fakeddit data scripts
# image_files = glob.glob("./images_500/images/images/*.jpg")
# all_data = []
# with open("image_info.csv", 'r') as inf:
#     data = csv.reader(inf)
#     next(data)
#     for line in data:
#         text = line[6]
#         image_id = line[10]
#         label = line[18] # 0-true, 1-fake
#         d = {}
#
#         if len(text) > 10 and image_id + ".jpg" in [name.split("/")[-1] for name in image_files]:
#             d["id"] = image_id
#             d["txt"] = text
#             d['label'] = label
#         else:
#             continue
#         all_data.append(d)

## fakenewsnet data scripts
image_files = glob.glob("../FakeNewsNet-master/code/fakenewsnet_dataset/politifact_multi/poli_img_all/*.jpg")
all_data = []
with open('../FakeNewsNet-master/politifact_multi.csv','r') as inf:
    data = csv.reader(inf)
    next(data)
    for line in data:
        text = line[1]
        image_id = line[2]
        label = line[3] # 0-true, 1-fake
        d = {}

        if len(text) > 0 and image_id + ".jpg" in [name.split("/")[-1] for name in image_files]:
            d["id"] = image_id
            d["txt"] = text
            d['label'] = int(label)
        else:
            continue
        all_data.append(d)


device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

dataset = []
for idx, d in enumerate(all_data):
    input_example = InputExample(text_a=d['txt'], label=int(d['label']), guid=d['id'])
    dataset.append(input_example)

#
sampler = FewShotSampler(num_examples_per_label=1, num_examples_per_label_dev=1, also_sample_dev=True)

train, dev = sampler.__call__(train_dataset=dataset, seed=3)

test = test_set(dataset, train, dev)

plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "roberta-base")

mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer,
                            text='{"soft":"<head>"}Here is a piece of news with {"mask"} information.{"soft":"<tail>"} {"placeholder":"text_a"} {"placeholder":"text_b"}')

train_dataloader = PromptDataLoader(dataset=train, template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
    batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="head")

validation_dataloader = PromptDataLoader(dataset=dev, template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
        batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
        truncate_method="head")

test_dataloader = PromptDataLoader(dataset=test, template=mytemplate, tokenizer=tokenizer,
                                   tokenizer_wrapper_class=WrapperClass, max_seq_length=512, decoder_max_length=3,
                                   batch_size=4, shuffle=True, teacher_forcing=False, predict_eos_token=False,
                                   truncate_method="head")


myverbalizer = SoftVerbalizer(tokenizer, plm, num_classes=2)

use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False)
if use_cuda:
    prompt_model = prompt_model.cuda()

loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.01},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
     'weight_decay': 0.0}
]
# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    {'params': prompt_model.verbalizer.group_parameters_1, "lr": 3e-5},
    {'params': prompt_model.verbalizer.group_parameters_2, "lr": 3e-4},
]
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

class Alpha(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_beta = torch.nn.Parameter(data=torch.Tensor(0), requires_grad=True)

    def forward(self):  # no inputs
        beta = torch.sigmoid(self.raw_beta)  # get (0,1) value
        return beta


class Proj_layers(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj1 = torch.nn.Linear(1024, 768, device=device)
        self.ln1 = torch.nn.LayerNorm(768, device=device)
        self.proj2 = torch.nn.Linear(768, 768, device=device)
        self.ln2 = torch.nn.LayerNorm(768, device=device)
        # self.sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.sig = torch.nn.Sigmoid()

    def forward(self, txt, img):
        out_emb = torch.cat((img, txt), 1)
        out_emb = self.ln1(F.relu(self.proj1(out_emb.float())))
        out_emb = self.ln2(F.relu(self.proj2(out_emb)))
        return out_emb

proj = Proj_layers()

def mini_batching(inputs):
    mini_batch = []
    sim_all = []
    sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    sig = torch.nn.Sigmoid()
    for item in inputs['guid']:
        for sample in all_data:
            if item == sample['id']:
                i_input = preprocess(Image.open("../FakeNewsNet-master/code/fakenewsnet_dataset/politifact_multi/poli_img_all/" + item + ".jpg")).unsqueeze(0).to(device)
                t_input = clip.tokenize(sample['txt'], truncate=True).to(device)

                i_emb = model.encode_image(i_input)
                t_emb = model.encode_text(t_input)
                # out_emb = t_emb + i_emb
                # proj = torch.nn.Linear(512, 768, device=device)
                out_emb = proj(t_emb, i_emb)

                sim_all.append(sim(t_emb, i_emb))

                # out_emb = torch.cat((t_emb, i_emb),-1)
                mini_batch.append(out_emb)
    sim_all = torch.stack(sim_all).to(device).squeeze()
    mini_batch = torch.stack(mini_batch).to(device).squeeze()
    sim_mean = torch.mean(sim_all)
    sim_std = torch.std(sim_all)
    normalized_mini = sig((mini_batch - sim_mean) / sim_std)
    mini_batch = normalized_mini * mini_batch
    return mini_batch





def train(model, train_dataloader, val_dataloader, test_dataloader, epoch, loss_function, optimizer, alpha=None):

    saved_model = None
    val_f1_macro_in_alpha = 0

    for epoch in range(epoch):
        tot_loss = 0
        print("===========EPOCH:{}=============".format(epoch+1))
        for step, inputs in enumerate(train_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            out = model.forward_without_verbalize(inputs)
            mini_batch = mini_batching(inputs)
            out = alpha * mini_batch + out
            logits = model.verbalizer.process_outputs(outputs=out, batch=inputs)
            # logits = prompt_model(inputs)
            labels = inputs['label']
            loss = loss_function(logits, labels)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
            print(tot_loss/(step+1))

        model.eval()

        allpreds = []
        alllabels = []
        for step, inputs in enumerate(val_dataloader):
            if use_cuda:
                inputs = inputs.cuda()
            out = model.forward_without_verbalize(inputs)

            mini_batch = mini_batching(inputs)
            out = alpha * out + mini_batch

            logits = model.verbalizer.process_outputs(outputs=out, batch=inputs)
            labels = inputs['label']
            alllabels.extend(labels.cpu().tolist())
            allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
        print("validation:",acc)
        report_val = classification_report(alllabels, allpreds, output_dict=True,
                                           labels=[0,1], target_names=["real", "fake"],
                                           )
        f1_fake = report_val['fake']['f1-score']
        f1_real = report_val['real']['f1-score']
        f1_macro = report_val['macro avg']['f1-score']
        if float(f1_macro) > val_f1_macro_in_alpha:
            val_f1_macro_in_alpha = float(f1_macro)
            print("saving model at {} alpha with {} f1 score".format(alpha, f1_macro))
            saved_model = model

    allpreds = []
    alllabels = []
    for step, inputs in enumerate(test_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        out = saved_model.forward_without_verbalize(inputs)

        mini_batch = mini_batching(inputs)
            # mini_batch = torch.stack(mini_batch).to(device).squeeze(1)
        out = alpha * out + mini_batch
            # out = torch.cat((mini_batch, out), -1)
            # proj = torch.nn.Linear(1792, 768, device=device)
            # out = proj(out)
        # # print(out.size()) #[1,768]
        logits = saved_model.verbalizer.process_outputs(outputs=out, batch=inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
    acc = sum([int(i==j) for i,j in zip(allpreds, alllabels)])/len(allpreds)
    print("test:", acc)
    report_test = classification_report(alllabels, allpreds, labels=[0,1], target_names=["real", "fake"])
    print(report_test)

alpha = 1
train(prompt_model,train_dataloader,validation_dataloader,test_dataloader,
          epoch=20,loss_function=loss_func,optimizer=optimizer1, alpha=alpha)