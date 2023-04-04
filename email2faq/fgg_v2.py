#%%
import cohere
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

CO = cohere.Client()  # put your cohere API trial key here


def mbkmeans_clusters(
    X,
    k,
    mb,
    print_silhouette_values,
    print_stats = True
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    if(print_stats):
        print(f"For n_clusters = {k}")
        print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
        print(f"Inertia:{km.inertia_}")

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ ==
                                                                 i]
            silhouette_values.append((
                i,
                cluster_silhouette_values.shape[0],
                cluster_silhouette_values.mean(),
                cluster_silhouette_values.min(),
                cluster_silhouette_values.max(),
            ))
        silhouette_values = sorted(silhouette_values,
                                   key=lambda tup: tup[2],
                                   reverse=True)
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_, k, silhouette_score(X, km.labels_)


def get_clusters(valid_questions, frequency=2, k=None):
    # get embeddings from cohere
    response = CO.embed(model='large', texts=valid_questions)
    embed_array = np.asarray(response.embeddings)

    if k and k>=2:
        required_k_value = k
    else:
        # get optimal k value
        max_clusters = len(embed_array) // 2
        k_values = []
        silhouette_values = []
        print("check for optimal k value")
        for k in tqdm(range(2, max_clusters)):
            _, _, k_value, silhouette_value = mbkmeans_clusters(
                X=embed_array,
                k=k,
                mb=5,
                print_silhouette_values=False,
                print_stats=False,)
            k_values.append(k_value)
            silhouette_values.append(silhouette_value)
        required_k_value = k_values[silhouette_values.index(max(silhouette_values))]
    
    # get clusters
    _, cluster_labels, _, _ = mbkmeans_clusters(
    	X=embed_array,
        k=required_k_value,
        mb=10,
        print_silhouette_values=True,
    )
    df_clusters = pd.DataFrame({
        "text": valid_questions,
        "cluster": cluster_labels
    })

    df_clusters_grp = df_clusters.groupby(["cluster"]).groups.values()
    clusters = []
    for cluster_id in range(len(df_clusters_grp)):
        cluster = df_clusters.iloc[list(list(df_clusters_grp)[cluster_id])]['text'].to_list()
        if len(cluster) >= frequency:
            clusters.append(cluster)
    
    return clusters

# #%%
# df = pd.read_csv("/home/aryanab/IEEE-year-long-project/email-2-faq/email2faq/data/asset/queries_sample.csv")
# valid_questions = df['sentences'].values.tolist()
# #%%
# clusters = get_clusters(valid_questions, frequency=2)

# print(len(clusters))
# # %%
# clusters
# # %%
