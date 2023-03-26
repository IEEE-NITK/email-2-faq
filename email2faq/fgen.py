#%%
import torch
from typing import List
import pandas as pd
from data.dataset import EmailsDataset, QueryDataset
from data.dataloader import QCDataLoader, FGGDataLoader
import qc
import fgg
#%%
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import sentencepiece
from data.dataset import ClusterDataset
from data.dataloader import FGENDataLoader
#%%
PATH = "/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/sample_email_dataset_raw.csv"
THRESHOLD = 0.4
FREQUENCY = 0 # TO BE MORE THAN 1

device = torch.device("cpu")
#%%
dataset = EmailsDataset(PATH)
loader = QCDataLoader(dataset)

valid_questions: pd.DataFrame = qc.get_valid_questions(loader, device)
#%%
fgg_dataset = QueryDataset(valid_questions)
fgg_loader = FGGDataLoader(fgg_dataset)

similar_query_pairs: pd.DataFrame = fgg.get_similarities(fgg_loader, threshold=THRESHOLD)
#%%
query_clusters: List = fgg.get_clusters(fgg_loader, similar_query_pairs, frequency=FREQUENCY)

# %%
data = ClusterDataset(query_clusters)
print(data.query_para_clusters)
loader = FGENDataLoader(data).dataloader

#%%
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

#%%
def summarizer(loader,device,model):
    model.to(device)
    faq : List = []
    inputs = [tokenizer(text, max_length=1024, padding = True , truncation =True, return_tensors="pt") for text in loader]
    for input in inputs:
      input.to(device)
      summary_id = model.generate(input["input_ids"]).to(device)
      faq.append(tokenizer.batch_decode(summary_id, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return faq
#%%
faq_clusters: List = summarizer(loader = loader,device = device , model = model)
#print(faq_clusters)