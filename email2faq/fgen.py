#%%
import torch
from typing import List, Dict
import pandas as pd
from data.dataset import EmailsDataset, QueryDataset, ClusterDataset
from data.dataloader import QCDataLoader, FGGDataLoader, FGDataLoader
import qc
import fgg
import fgg_v2
import fg
import logging

logging.basicConfig(level=logging.INFO)


def generate_faq_fgen(path: str,
                      threshold: float = 0.35,
                      frequency: int = 1) -> Dict:
    device = torch.device("cpu")

    # QC subsystem
    dataset = EmailsDataset(path)
    loader = QCDataLoader(dataset)
    valid_questions: pd.DataFrame = qc.get_valid_questions(loader, device)
    logging.info('Query Classification (QC) completed!')

    # # FGG subsystem
    # fgg_dataset = QueryDataset(valid_questions)
    # fgg_loader = FGGDataLoader(fgg_dataset)
    # similar_query_pairs: pd.DataFrame = fgg.get_similarities(
    #     fgg_loader, threshold=threshold)
    # query_clusters: List = fgg.get_clusters(fgg_loader,
    #                                         similar_query_pairs,
    #                                         frequency=frequency)
    # logging.info('FAQ Group Generator (FGG) completed!')

    # FGG subsystem v2 (k-means based)
    valid_questions_list = valid_questions['sentences'].values.tolist()
    query_clusters: List = fgg_v2.get_clusters(valid_questions_list, frequency=frequency)
    logging.info('FAQ Group Generator (FGG) v2 completed!')

    # FG subsystem
    fg_dataset = ClusterDataset(query_clusters)
    fg_loader = FGDataLoader(fg_dataset)
    faq: List = fg.generate_faq(fg_loader, device)
    valid_faq = []
    for i in range(len(faq)):
        if 'All photographs are copyrighted.' in faq[i][0]:
            continue
        valid_faq.append(faq[i][0])
    logging.info('FAQ Generator (FG) completed!')

    result = {
        "valid_queries": valid_questions['sentences'].values.tolist(),
        "query_clusters": query_clusters,
        "valid_faq": valid_faq
    }
    return result
#%%
# generate_faq_fgen("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/sample_email_dataset_raw.csv")
#%%

# #%%
# PATH = "/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/sample_email_dataset_raw.csv"
# THRESHOLD = 0.3
# FREQUENCY = 1  # TO BE MORE THAN 1

# device = torch.device("cpu")
# #%%
# dataset = EmailsDataset(PATH)
# loader = QCDataLoader(dataset)

# valid_questions: pd.DataFrame = qc.get_valid_questions(loader, device)
# #%%
# fgg_dataset = QueryDataset(valid_questions)
# fgg_loader = FGGDataLoader(fgg_dataset)

# similar_query_pairs: pd.DataFrame = fgg.get_similarities(fgg_loader,
#                                                          threshold=THRESHOLD)
# #%%
# query_clusters: List = fgg.get_clusters(fgg_loader,
#                                         similar_query_pairs,
#                                         frequency=FREQUENCY)

# # %%
# fg_dataset = ClusterDataset(query_clusters)
# fg_loader = FGDataLoader(fg_dataset)

# #%%
# faq: List = fg.generate_faq(fg_loader, device)
# #%%
# valid_faq = []
# for i in range(len(faq)):
#     if 'All photographs are copyrighted.' in faq[i][0]:
#         continue
#     valid_faq.append(faq[i][0])
# print(valid_faq)
# # %%

# %%
