#%%
import os
import pandas as pd
from torch.utils.data import Dataset
from utils.preprocess_utils_qc import email_preprocess

class EmailsDataset(Dataset):
    """
    """
    def __init__(self, emails_file, email_field = "mails"):
        self.emails = pd.read_csv(emails_file)
        self.email_field = email_field
        self._preprocess_emails()

    def __len__(self):
        return len(self.sentences)

    def _preprocess_emails(self):
        self.sentences, self.sentences_field = email_preprocess(self.emails, self.email_field)

    def __getitem__(self, idx):
        sentence = self.sentences[self.sentences_field][idx]
        return sentence
#%%
# pd.read_csv("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/sample_email_dataset_raw.csv")
# #%%
# k = EmailsDataset("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/sample_email_dataset_raw.csv")
# k.__getitem__(2)
# %%
