import numpy as np 
import pandas as pd
import re
import email

def email_preprocess(emails_df, email_field):
    """
    """
    # Parse the emails into a list email objects
    messages = list(map(email.message_from_string, emails_df[email_field]))
    emails_df.drop(email_field, axis=1, inplace=True)
    
    # Get fields from parsed email objects
    keys = messages[0].keys()
    for key in keys:
        emails_df[key] = [doc[key] for doc in messages]
    
    # Parse content from emails
    emails_df['content'] = list(map(get_text_from_email, messages))
    
    # Clean parsed emails
    emails_df['content_clean'] = emails_df['content'].apply(clean_email_column)

    # Split sentences and create new dataframe
    sentences = []
    emails_df['content_clean'].apply(lambda x: sentences.extend(split_sentence(x)))
    sentences_df = pd.DataFrame(sentences)
    sentences_df.rename(columns = {0:"sentences"}, inplace = True)
    return sentences_df, sentences_df.columns[0]


def get_text_from_email(msg):
    '''To get the content from email objects'''
    parts = []
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            parts.append( part.get_payload() )
    return ''.join(parts)

def clean_email_column(data):
    if data is not None:
        # stopwords_list = stopwords.words('english')
        # exclusions = ['RE:', 'Re:', 're:']
        # exclusions = '|'.join(exclusions)
        data =  data.lower()
        data = re.sub('re:', '', data)
        data = re.sub('-', '', data)
        data = re.sub('_', '', data)
        # Remove data between square brackets
        data =re.sub('\[[^]]*\]', '', data)
        # removes punctuation
        # data = re.sub(r'[^\w\s]','',data)
        data = re.sub(r'\n',' ',data)
        data = re.sub(r'\t',' ',data)
        data = re.sub(r'[0-9]+','',data)
        # strip html 
        p = re.compile(r'<.*?>')
        data = re.sub(r"\'ve", " have ", data)
        data = re.sub(r"can't", "cannot ", data)
        data = re.sub(r"n't", " not ", data)
        data = re.sub(r"I'm", "I am", data)
        data = re.sub(r" m ", " am ", data)
        data = re.sub(r"\'re", " are ", data)
        data = re.sub(r"\'d", " would ", data)
        data = re.sub(r"\'ll", " will ", data)
        data = re.sub('forwarded by phillip k allenhouect on    pm', '',data)
        data = re.sub(r"httpitcappscorpenroncomsrrsauthemaillinkaspidpage", "", data)
        
        data = p.sub('', data)
        # if 'forwarded by:' in data:
        if 'forwarded by ' in data:
            data = data.split('forwarded by ')[0]
        if ('to:' in data) and ("subject:" in data):
            data = data.split('to:')[0]
        data = data.strip()
        return data
    return 'No Subject'

def split_sentence(text):
  sentences_list = re.split(r"[.;!?]\s+", text)
  return sentences_list

#%%
# import re
# text = """Though Hertzberg was initially a supporter, his legal consultant has been 
# scared off by the Gov's people (i.e., to de-link would ""impair"" the 
# contracts, triggering breach, liquidated damages and the end of Western 
# Civilization as we know it).  The Senate ain't buying that line."""
# sentences_list = re.split(r"[.;!?]\s+", text)
# print(sentences_list)

# %%
