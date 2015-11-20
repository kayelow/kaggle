import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import json
from pprint import pprint
from collections import defaultdict, Counter
import pandas as pd

recipe_cuisine_ingredient = json.load(open('train.json'))
pprint(recipe_cuisine_ingredient[0])

unique_cuisines = list(set(recipe['cuisine'] for recipe in recipe_cuisine_ingredient))
pprint(unique_cuisines)
print(len(unique_cuisines))

cuisines_ingredients = {recipe['cuisine']:[] for recipe in recipe_cuisine_ingredient}
for recipe in recipe_cuisine_ingredient: 
    cuisines_ingredients[recipe['cuisine']].extend(recipe['ingredients'])

cuisines_ingredients_count = {k:Counter(v) for k,v in cuisines_ingredients.iteritems()}
# pprint(cuisines_ingredients_count['irish']['salt'])

unique_ingredients = list(set(' '.join(recipe['ingredients']) for recipe in recipe_cuisine_ingredient))
print(len(unique_ingredients))

# preview our df
recipes_df = pd.DataFrame.from_dict(recipe_cuisine_ingredient,orient='columns')
print(recipes_df.ix[:10][:])

# number of recipes per cuisine
cuisines_s = recipes_df.ix[:,0]
recipes_per_cuisine = pd.DataFrame(cuisines_s.value_counts())

cuisine_ingredients = recipes_df[['cuisine','ingredients']]
print(cuisine_ingredients.ix[:10][:])

cuisines = recipes_df[['cuisine']]
ingredients = recipes_df[['ingredients']]
ci_matrix = pd.DataFrame.as_matrix(ingredients)
print ci_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
# http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#bags-of-words

recipes_as_documents = [' '.join(recipe['ingredients']) for recipe in recipe_cuisine_ingredient]
cuisine_classification = [recipe['cuisine'] for recipe in recipe_cuisine_ingredient]

# occurances 
count_vect = CountVectorizer(decode_error='strict')
counts_ingredients = count_vect.fit_transform(recipes_as_documents)
counts_ingredients.shape

# frequencies

tfidf_transformer = TfidfTransformer()
ingredient_tfidf = tfidf_transformer.fit_transform(counts_ingredients)
ingredient_tfidf.shape

classifier_of_cuisines = MultinomialNB().fit(ingredient_tfidf, cuisine_classification)
classifier_of_cuisines = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
                    ])

classifier_of_cuisines = classifier_of_cuisines.fit(recipes_as_documents, cuisine_classification)

# test naive bayes
# results in score of 0.68111
test_data = json.load(open('test.json'))
test_recipes = [' '.join(recipe['ingredients']) for recipe in test_data]
predictions = classifier_of_cuisines.predict(test_recipes)

pprint(predictions.tolist()[:10])

prediction_df = pd.DataFrame({'id': [recipe['id'] for recipe in test_data], 'cuisine': predictions.tolist()})
prediction_df = prediction_df[['id','cuisine']]

import matplotlib
prediction_df['cuisine'].value_counts().plot(kind='bar', alpha=0.5)

# prediction to CSV
prediction_df.to_csv('submission.csv', sep=',', index=False)

# prediction to CSV
prediction_df.to_csv('submission.csv', sep=',', index=False)

# SVD
from sklearn.decomposition import TruncatedSVD

# lets start with n = 2
svd = TruncatedSVD(n_components=2)
svd_fit = svd.fit(ingredient_tfidf)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_)

# n = 10
svd = TruncatedSVD(n_components=10)
svd_fit = svd.fit(ingredient_tfidf)
print(svd.explained_variance_ratio_)
print(svd.explained_variance_)

from sklearn.cluster import KMeans, MiniBatchKMeans

# the default is n = 8; we have 20 cusines 
km = MiniBatchKMeans(n_clusters=20, init='k-means++', n_init=1,
                         init_size=1000, batch_size=1000, verbose=True)

km.fit(ingredient_tfidf) 

order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = count_vect.get_feature_names()
for i in xrange(20):
    print("Cluster %d:" % i)
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
    print()

# try a different vectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf_vectorizer = TfidfVectorizer(max_df=.5, max_features=39674,
                                 min_df=.1,
                                 use_idf=True, ngram_range=(1,3))
ing_fit = tf_idf_vectorizer.fit_transform(recipes_as_documents)
svd = TruncatedSVD(20)
normalizer = Normalizer(copy=False)
lsa = make_pipeline(svd, normalizer)

ing_fit = lsa.fit_transform(ing_fit)
print(svd.explained_variance_ratio_.sum())

km_2 = KMeans(n_clusters=20, init='k-means++', max_iter=100, n_init=1,
                verbose=False)
km_2.fit(ing_fit)
original_space_centroids_2 = svd.inverse_transform(km_2.cluster_centers_)
order_centroids_2 = original_space_centroids_2.argsort()[:, ::-1]
terms = tf_idf_vectorizer.get_feature_names()
for i in xrange(20):
    print("Cluster %d:" % i)
    for ind in order_centroids_2[i, :30]:
        print(' %s' % terms[ind])
    print()

classifier_of_cuisines_2 = MultinomialNB().fit(ingredient_tfidf, cuisine_classification)
classifier_of_cuisines_2 = Pipeline([('vect', TfidfVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
                    ])

classifier_of_cuisines_2 = classifier_of_cuisines_2.fit(recipes_as_documents, cuisine_classification)
test_data = json.load(open('test.json'))
test_recipes = [' '.join(recipe['ingredients']) for recipe in test_data]
predictions = classifier_of_cuisines_2.predict(test_recipes)

pprint(predictions.tolist()[:10])

prediction_df = pd.DataFrame({'id': [recipe['id'] for recipe in test_data], 'cuisine': predictions.tolist()})
prediction_df = prediction_df[['id','cuisine']]
import matplotlib
prediction_df['cuisine'].value_counts().plot(kind='bar', alpha=0.5)

# prediction to CSV
prediction_df.to_csv('submission_2.csv', sep=',', index=False)
