#%%
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/question-vs-statement-classifier")
# model.save_pretrained("./models_store/qc_model/")

MODEL = AutoModelForSequenceClassification.from_pretrained(
    "./models_store/qc_model/")


def get_valid_questions(loader, device, model=MODEL):
    model.to(device)
    valid_questions_idx = []
    batch_size = loader.batch_size
    adder = 0
    ## evaluate data for one epoch
    for batch in tqdm(loader.dataloader):
        ## add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        ## unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        ## avoiding model's computation and storage of gradients -> saving memory and speeding up validation
        with torch.no_grad():
            # forward pass, calculate logit predictions
            logits = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask)

        ## move logits and labels to CPU
        logits = logits['logits'].detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()
        # print(preds)
        valid_questions_idx.extend((np.where(preds == 1)[0] + adder).tolist())
        adder += batch_size
    valid_questions = loader.dataset.sentences.iloc[valid_questions_idx]
    return valid_questions.reset_index(drop=True)


# #%%
# from data.dataset import EmailsDataset
# from data.dataloader import QCDataLoader
# dataset = EmailsDataset("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/sample_email_dataset_raw.csv")
# loader = QCDataLoader(dataset)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# valid_questions = get_valid_questions(loader, device)

# %%
