from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from os.path import join

n_top_words = 20


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


dir_path = "/Users/tnaghibi/Downloads/OLX/olx_data_sample/"
data = pd.read_csv("/Users/tnaghibi/Downloads/OLX/olx_data_sample/za_sample_listings_incl_cat.csv")
uniq_l2 = filter(lambda v: v == v, data["category_l2_name_en"].unique())
data["CLUSTER_ID"] = 0
perplexity_scores = []
for cluster_ind, l2_cat in enumerate(uniq_l2):
    data_l2 = data.loc[data["category_l2_name_en"] == l2_cat]
    stopwords = {".", "|", "the", "s", "with", "free", "delivery", "", "!", "!!", "\""}
    documents = map(lambda title: ' '.join(
        map(lambda word: word.strip(".").strip(","), filter(lambda word: word not in stopwords, title.lower()
                                                            .replace("\"", "")
                                                            .replace("\'", "")
                                                            .replace("&", " ")
                                                            .replace("!", "")
                                                            .replace("(", " ")
                                                            .replace(")", "")
                                                            .replace("-", "")
                                                            .replace(",", "")
                                                            .replace("/", "")
                                                            .split(" ")))),
                    data_l2["listing_title"])

    # Use tf (raw term count) features for LDA.
    print("Extracting tf features for LDA...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=1,
                                    max_features=None,
                                    analyzer='char_wb', ngram_range=(3, 3),
                                    stop_words='english')
    t0 = time()
    tf = tf_vectorizer.fit_transform(documents)
    print("done in %0.3fs." % (time() - t0))
    print()

    n_features = tf.shape[1]
    n_samples = tf.shape[0]
    n_components = min(int(n_samples / 10.0),1000)

    print("Fitting LDA models with tf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    print("\nClusters in LDA model:")

    cluster_ids = np.argmax(lda.transform(tf), axis=1)

    data.loc[data["category_l2_name_en"] == l2_cat]["CLUSTER_ID"] = np.array(
        [str(cluster_ind) + str(id) for id in cluster_ids])
    score = lda.perplexity(tf)
    perplexity_scores.append(score)
    print("Done with l2 #%d,l2 is %s" % (cluster_ind, l2_cat))

print ("Perplexity Score over l2 categories: ")
print (perplexity_scores)

data.to_csv(join(dir_path, 'za_sample_listings_incl_cat_with_clusters.csv'), index=False)
