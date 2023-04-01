#%%
import numpy as np
import pandas as pd
import tensorflow as tf
from utils.similarity import ManDist
from utils.clustering import similar_groups
from tqdm import tqdm

# MODEL = tf.keras.models.load_model("./models_store/fgg_model/SiameseLSTM.h5", custom_objects={'ManDist': ManDist})

def get_similarities(loader, threshold=0.4):
    model = tf.keras.models.load_model("./models_store/fgg_model/SiameseLSTM.h5", custom_objects={'ManDist': ManDist})
    # model.to(device)
    similarities_idx = []
    batch_size = loader.batch_size
    adder = 0
    ## evaluate data for one epoch
    for batch in tqdm(loader.dataloader):
        ## unpack the inputs from our dataloader
        b_left, b_right = batch
        b_left = b_left.numpy()
        b_right = b_right.numpy()
        prediction = model.predict([b_left, b_right])
        similarities_idx.extend((np.where(prediction.flatten() > threshold)[0] + adder).tolist())
        adder += batch_size
    similar_query_pairs = loader.dataset.pairwise_idx_df.iloc[similarities_idx]
    del model
    return similar_query_pairs.reset_index(drop=True)

# %%
# create clusters of similar pairs
def get_clusters(loader, similar_query_pairs, frequency=1):
    groups = similar_groups(similar_query_pairs)
    clusters = []
    for group in groups:
        cluster = loader.dataset.query_df.iloc[group][loader.dataset.query_field].to_list()
        if len(cluster) > frequency:
            clusters.append(cluster)
    return clusters

# #%%
# from data.dataset import QueryDataset
# from data.dataloader import FGGDataLoader
# from tqdm import tqdm
# import pandas as pd

# df = pd.read_csv("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/queries_sample.csv")
# dataset = QueryDataset(df)
# loader = FGGDataLoader(dataset)

# df = get_similarities(loader)
# get_clusters(loader, df)
# # %%
