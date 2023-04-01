#%%
import pandas as pd
from torch.utils.data import Dataset
from data.utils.preprocess_utils_qc import email_preprocess
from data.utils.preprocess_utils_fgg import make_w2v_embeddings, split_and_zero_padding
from keras_preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer
from transformers import PegasusTokenizer
from itertools import product

#%%
# tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/question-vs-statement-classifier")
# tokenizer.save_pretrained("./utils/tokenizers/")

#%%
# TOKENIZER = AutoTokenizer.from_pretrained("./data/utils/tokenizers/")
# FG_TOKENIZER = PegasusTokenizer.from_pretrained("./data/utils/summ_tokenizers/")
#%%
class EmailsDataset(Dataset):
    """
    """
    def __init__(self, emails_file, email_field = "mails"):
        self.emails = pd.read_csv(emails_file)
        self.email_field = email_field
        self._preprocess_emails()

        # transformer preprocess parameters
        self.tokenizer = AutoTokenizer.from_pretrained("./data/utils/tokenizers/")
        self.max_sequence_length = 512

    def __len__(self):
        return len(self.sentences)

    def _preprocess_emails(self):
        self.sentences, self.sentences_field = email_preprocess(self.emails, self.email_field)

    def __getitem__(self, idx):
        sentence = self.sentences[self.sentences_field][idx]
        return sentence
    
    def _preprocess(self, texts):
        """
        Add bert token (CLS and SEP) tokens to each sequence pre-tokenization
        """
        ## separate labels and texts before preprocessing
        # Adding CLS and SEP tokens at the beginning and end of each sequence for BERT
        texts_processed = ["[CLS] " + str(sequence) + " [SEP]" for sequence in texts]
        return texts_processed
    
    def _tokenize(self, texts):
        """
        Use bert tokenizer to tokenize each sequence and post-process 
        by padding or truncating to a fixed length
        """
        ## tokenize sequence
        tokenized_texts = [self.tokenizer.tokenize(text) for text in (texts)]
        
        ## convert tokens to ids
        text_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in (tokenized_texts)]

        ## pad our text tokens for each sequence
        text_ids_post_processed = pad_sequences(text_ids, 
                                       maxlen=self.max_sequence_length, 
                                       dtype="long", 
                                       truncating="post", 
                                       padding="post") 
        return text_ids_post_processed

    def _create_attention_mask(self, text_ids):
        """
        Add attention mask for padding tokens
        """
        attention_masks = []
        # create a mask of 1s for each token followed by 0s for padding
        for seq in (text_ids):
            seq_mask = [float(i>0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks
    
    def process_texts(self):
        """
        Apply preprocessing and tokenization pipeline of texts
        """

        ## perform the split
        texts = self.sentences[self.sentences_field].values

        ## preprocess train, val, test texts
        test_texts_processed = self._preprocess(texts)
        
        del texts
        
        ## preprocess train, val, test texts
        ids = self._tokenize(test_texts_processed)

       
        del test_texts_processed 
        
        ## create masks for train, val, test texts
        masks = self._create_attention_mask(ids)
        del self.tokenizer
        return ids, masks

#%%
# pd.read_csv("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/sample_email_dataset_raw.csv")
# #%%
# k = EmailsDataset("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/sample_email_dataset_raw.csv")
# k.__getitem__(2)
# %%

class QueryDataset(Dataset):
    """
    """
    def __init__(self, query_df):
        self.query_df = query_df
        self.query_field = self.query_df.columns[-1]
        self.embedding_dim = 300
        self.max_seq_length = 20
        self._pairwise()

    def __len__(self):
        return len(self.query_df)

    def _pairwise(self):
        self.pairwise_df = pd.DataFrame(product(self.query_df[self.query_field], self.query_df[self.query_field]))
        self.pairwise_df = self.pairwise_df.rename(columns={0:'question1', 1:'question2'})
        for q in ['question1', 'question2']:
            self.pairwise_df[q + '_n'] = self.pairwise_df[q]
        
        # pairwise index
        self.query_df = self.query_df.loc[:, ~self.query_df.columns.str.match('Unnamed')].reset_index()
        self.pairwise_idx_df = pd.DataFrame(product(self.query_df["index"], self.query_df["index"]))

    def preprocess(self):
        self.pairwise_df, embeddings = make_w2v_embeddings(self.pairwise_df, embedding_dim = self.embedding_dim)
        return split_and_zero_padding(self.pairwise_df, self.max_seq_length)

    def __getitem__(self, idx):
        return self.query_df[self.query_field][idx]
    
    
# #%%
# df = pd.read_csv("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/queries_sample.csv")
# dataset = QueryDataset(df)
# dataset.preprocess()
# # %%

# class PairwiseDataset(Dataset):
#     """
#     """
#     def __init__(self, embeded_pairwise_df):
#         self.embeded_pairwise_df = embeded_pairwise_df

#     def __len__(self):
#         return len(self.embeded_pairwise_df)

#     def __getitem__(self, idx):
#         return [self.embeded_pairwise_df['left'][idx], self.embeded_pairwise_df['right'][idx]]
#%%

class ClusterDataset(Dataset):
    """
    Assuming that the data is in the form of a list of lists
    where each list contains a set of strings of similar sentences
    """

    def __init__(self, query_clusters):
        self.query_clusters = query_clusters
        self.sentences = [sen for cluster in self.query_clusters for sen in cluster]
        self.query_para_clusters = [("? ".join(cluster) + "?") for cluster in self.query_clusters]

        self.tokenizer = PegasusTokenizer.from_pretrained("./data/utils/summ_tokenizers/")
    
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
        return self.sentences[idx]
        
    
    def sen_len(self):
        """
        Total number of sentences present in the query_clusters
        """
        return len(self.sentences)
    
    def preprocess(self):
        inputs = [self.tokenizer(text, max_length=1024, padding = True , truncation =True, return_tensors="pt") for text in self.query_para_clusters]
        return inputs

# #%%
# query_clusters = [ ["What are the different types of laptops available?",
# "What are the specifications of each type of laptop?",
# "What is the battery life of each laptop model?",
# "What is the warranty period of the laptops?",
# "What is the cost of each laptop model?"],
# [
# "What destinations are available for travel?",
# "What is the duration of the trips?",
# "What is the cost of the trips?",
# "What are the payment options available?",
# "Is there an option for installment payment?"]]
# dataset = ClusterDataset(query_clusters)
# inputs_t = dataset.preprocess()
# # %%

