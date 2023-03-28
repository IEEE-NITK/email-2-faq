import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import pandas as pd
from typing import List


# import gensim
# from gensim.summarization import summarize
# from transformers import PegasusForConditionalGeneration, PegasusTokenizer

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

query_clusters: List = [ ["What are the different types of laptops available?",
"What are the specifications of each type of laptop?",
"What is the battery life of each laptop model?",
"What is the warranty period of the laptops?",
"What is the cost of each laptop model?",
"Can I customize the laptop according to my needs?",
"What are the available payment options?",
"Is there an option for installment payment?",
"How can I check the status of my order?",
"What is the estimated delivery time for the laptop?",
"Is there an option for expedited delivery?",
"Can I return the laptop if I am not satisfied with it?",
"What is the return policy of the company?",
"Will I get a refund if I return the laptop?",
"Is there an option for exchange?",
"What is the process for laptop repair?",
"What is the cost of laptop repair?",
"What is the turnaround time for laptop repair?",
"Are there any additional charges for laptop repair?",
"What are the software options available on the laptop?",
"What are the pre-installed software on the laptop?",
"Can I install my own software on the laptop?",
"Are there any software compatibility issues?",
"What are the security features available on the laptop?",
"How can I protect my laptop from viruses?",
"What are the available antivirus software?",
"Is there a trial version of antivirus software available?",
"Can I install my own antivirus software on the laptop?",
"What are the data backup options available?",
"How can I backup my data on the laptop?",
"Is there a backup software available?",
"Is there an option for cloud backup?",
"What are the available storage options on the laptop?",
"Can I upgrade the storage on the laptop?",
"What is the maximum storage capacity of the laptop?",
"Are there any discounts available on the laptop?",
"Is there an option for bulk purchase?",
"What are the after-sales services available?",
"What is the technical support available?",
"What are the contact details for technical support?",
"Is there an online chat support available?",
"What are the operating systems available on the laptop?",
"What are the differences between the operating systems?",
"Can I switch to a different operating system?",
"Is there a cost for switching to a different operating system?",
"What are the language options available on the laptop?",
"Can I change the language settings?",
"Is there a user manual available?",
"What are the tips for laptop maintenance?",
"How can I prolong the life of my laptop?"],

[
"What destinations are available for travel?",
"What is the duration of the trips?",
"What is the cost of the trips?",
"What are the payment options available?",
"Is there an option for installment payment?",
"What is included in the package?",
"What are the additional costs that I need to bear?",
"Is there a discount available for group travel?",
"What is the maximum number of travelers in a group?",
"What is the minimum number of travelers required for group travel?",
"What is the minimum age limit for travel?",
"Are there any restrictions on the nationality of travelers?",
"What is the process for obtaining travel visas?",
"Is there assistance available for obtaining travel visas?",
"What is the estimated time for obtaining travel visas?",
"What is the validity period of the travel visas?",
"What are the accommodation options available?",
"What are the hotel ratings?",
"What are the facilities available in the hotels?",
"Is there an option for a private room?",
"Is there an option for sharing the room with other travelers?",
"What are the transportation options available?",
"Is transportation included in the package?",
"Is there an option for private transportation?",
"What is the type of transportation provided?",
"Is there an option for renting a car?",
"What are the attractions covered in the package?",
"Are there any additional attractions that can be included?",
"Is there a guide available for sightseeing?",
"Is there a language barrier for sightseeing?",
"What are the shopping options available?",
"Is there a shopping guide available?",
"What is the return policy for items purchased during the trip?",
"Is there an option for travel insurance?",
"What is covered in the travel insurance?",
"What is the cost of the travel insurance?",
"What is the process for making a claim on the travel insurance?",
"What are the food options available?",
"Is there an option for vegetarian or vegan food?",
"Are there any dietary restrictions to be followed?",
"What is the dress code for travel?",
"What are the weather conditions at the destination?",
"What are the health precautions to be taken during travel?",
"Are there any vaccinations required for travel?",
"Is there a medical facility available during travel?",
"What is the emergency contact information available?",
"Is there a local contact available for assistance?",
"What are the language options for communication?",
"Is there a customer service available during travel?",
"How can I provide feedback on the travel experience?"
]
]

class ClusterDataset(Dataset):
    """
    Assuming that the data is in the form of a list of lists
    where each list contains a set of strings of similar sentences
    """

    def __init__(self, query_clusters):
        self.query_clusters = query_clusters
        self.sentences = [sen for cluster in self.query_clusters for sen in cluster]
        self.query_para_clusters = [" ".join(cluster) for cluster in self.query_clusters]
    
    def __len__(self):
        """
        Total number of paragraphs present in the query_clusters
        """
        return len(self.query_para_clusters)
    
    def __getitem__(self, idx):
        """
        This is for accessing a paragraph of similar questions in a cluster
        """
        return self.query_para_clusters[idx]
    
    def sen_item(self,idx):
        """
        returns the idx_th sentence in the data set
        """
        return self.sentences[i]
        
    
    def sen_len(self):
        """
        Total number of sentences present in the query_clusters
        """
        return len(self.sentences)
    
    def preprocess():
        pass



class FGENDataLoader:
    ''' This class is written considering that input is a 
        list of lists where each list is contains the similar cluster of sentences '''

    '''Not sure how to deal with batches here. '''

    def __init__(self,data):
        self.data = data
        self.create_loader()

    def create_loader(self):
       sampler = SequentialSampler(data)
       self.dataloader = DataLoader(data,sampler=sampler)


data = ClusterDataset(query_clusters)
loader = FGENDataLoader(data).dataloader

""" def genism_summarizer(loader):
    faq : List = [summarize(text, word_count = 50) for text in loader]
    print(faq)
    return faq """

model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")

def summarizer(loader,device,model):
    model.to(device)
    faq : List = []
    inputs = [tokenizer(text, max_length=1024, padding = True , truncation =True, return_tensors="pt") for text in loader]
    for input in inputs:
      input.to(device)
      summary_id = model.generate(input["input_ids"]).to(device)
      faq.append(tokenizer.batch_decode(summary_id, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return faq



faq_clusters: List = summarizer(loader = loader,device = device , model = model)
print(faq_clusters)