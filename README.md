# Cross-Language-Tweets-Recommendation
Use latent semantic mapping and on-line machine learning algorithm to recommend tweets cross different languages.

The ultimate goal is to recommend relevant tweets irrespective of languages (users can define languages), facilitating bi/multi-lingual people to gain more useful infomation while search in twitter.

Build the machine learning pipeline, including raw stream tweets crawling, preprocessing, feature extraction, stochastic gradient descent(SGD) implementation to learn the latent mapping, and evaluation. 

The SGD training method is an on-line learning algorithm, which is suitable for incremental updating. I use SGD to learn a low-dimensional joint embedding space for both languages(English and Spanish). The intuitive explanation of this model is that it maps the tweets in different languages with the same hashtags into the same low dimensional space for computing their similarity, given the assumption that tweets sharing the same hashtag are likely to be topically relevant/similar. This model captures the latent semantics of different languages.

Reference:
SIGIR’14 Cross-language Context-Aware Citation Recommendation in Scientific Articles
