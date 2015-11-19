import json
from pprint import pprint
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

recipe_cuisine_ingredient = json.load(open('train.json'))
pprint(recipe_cuisine_ingredient[0])

unique_cuisines = list(set(
                    recipe['cuisine'] 
                    for recipe in recipe_cuisine_ingredient
                    ))
pprint(unique_cuisines)

cuisines_ingredients = {recipe['cuisine']:[] 
                        for recipe in recipe_cuisine_ingredient}

for recipe in recipe_cuisine_ingredient: 
    cuisines_ingredients[recipe['cuisine']].extend(recipe['ingredients'])

cuisines_ingredients_count = {
                                k:Counter(v) 
                                for k,v in cuisines_ingredients.iteritems()
                            }
# pprint(cuisines_ingredients_count['irish']['salt'])

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


recipes_as_documents = [
                        ' '.join(recipe['ingredients']) 
                        for recipe in recipe_cuisine_ingredient
                        ]
cuisine_classification = [
                        recipe['cuisine'] 
                        for recipe in recipe_cuisine_ingredient
                        ]

# occurances a la 
# ref -- http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
count_vect = CountVectorizer(decode_error='strict')
counts_ingredients = count_vect.fit_transform(recipes_as_documents)
counts_ingredients.shape

# frequencies

tfidf_transformer = TfidfTransformer()
ingredient_tfidf = tfidf_transformer.fit_transform(counts_ingredients)
ingredient_tfidf.shape

cuisine_clf = MultinomialNB().fit(ingredient_tfidf, cuisine_classification)
cuisine_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultinomialNB()),
                    ])

cuisine_clf = cuisine_clf.fit(recipes_as_documents, cuisine_classification)

# test naive bayes
# results in score of 0.68111
test_data = json.load(open('test.json'))
test_recipes = [' '.join(recipe['ingredients']) for recipe in test_data]
predictions = cuisine_clf.predict(test_recipes)

pprint(predictions.tolist()[:10])

prediction_df = pd.DataFrame({
                            'id': [recipe['id'] for recipe in test_data], 
                            'cuisine': predictions.tolist()
                            })
prediction_df = prediction_df[['id','cuisine']]

prediction_df['cuisine'].value_counts().hist()

prediction_df.to_csv('mn_nb_submission.csv', sep=',', index=False)