#%%
import torch
from typing import List
import pandas as pd
from data.dataset import EmailsDataset, QueryDataset
from data.dataloader import QCDataLoader, FGGDataLoader
import qc
import fgg
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
