#%%
import torch
import numpy as np
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration

# model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
# model.save_pretrained("./models_store/fg_model/")

# MODEL = PegasusForConditionalGeneration.from_pretrained("./models_store/fg_model/")

def generate_faq(loader, device):
    model = PegasusForConditionalGeneration.from_pretrained("./models_store/fg_model/")
    model.to(device)
    faq = []
    for input in tqdm(loader.inputs): # batching not implemented
      input.to(device)
      summary_id = model.generate(input["input_ids"]).to(device)
      faq.append(loader.dataset.tokenizer.batch_decode(summary_id, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    del loader.dataset.tokenizer
    del model
    return faq

# #%%
# from data.dataset import ClusterDataset
# from data.dataloader import FGDataLoader
# # query_clusters = [ ["What are the different types of laptops available?",
# # "What are the specifications of each type of laptop?",
# # "What is the battery life of each laptop model?",
# # "What is the warranty period of the laptops?",
# # "What is the cost of each laptop model?"],
# # [
# # "What destinations are available for travel?",
# # "What is the duration of the trips?",
# # "What is the cost of the trips?",
# # "What are the payment options available?",
# # "Is there an option for installment payment?"]]
# # query_clusters = [['what are you upto?'], ['is everythimg ok?']]
# dataset = ClusterDataset(query_clusters)
# loader = FGDataLoader(dataset)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #%%
# faq = generate_faq(loader, device)

# # %%
