install.packages("jsonlite")
install.packages("randomForest")
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
library("jsonlite")
library("randomForest")
library("rpart")

# load training data
recipes <- fromJSON("train.json",flatten=TRUE)
recipe_count <-length(recipes$id)

# what cuisines do we have?
list_of_cuisines<-unique(recipes$cuisine)

# how are the recipes distributed amongst the cuisines?
recipe_cuisine_counts <- table(recipes$cuisine)
barplot(recipe_cuisine_counts, main="Cuisine Distribution", ylab="Number of Recipes")

# ingredient count as a feature
recipes$ingredient_count <- lapply(recipes$ingredients, function(x) length(x))
recipes$ingredient_count <- unlist(recipes$ingredient_count)
ingredient_frequency <- table(unlist(recipes$ingredients))
barplot(ingredient_frequency, main="Ingredient Distribution", ylab="Number of Occurrances")

most_popular_ingredient <-names(which(ingredient_frequency==max(ingredient_frequency)))
infrequent_ingredients <- names(ingredient_frequency[ingredient_frequency >1000])
frequent_ingredients <- names(ingredient_frequency[ingredient_frequency >1000])
boxplot(ingredient_count~cuisine, data=recipes,main="Ingredient Counts per Cuisine")

# some ingredients are distinctly few cuisines, example: yuzu, fish sauce, soy sauce
# some ingredients span many cuisines, example: egg, bacon,tomato
recipes$yuzu <- grepl("yuzu", recipes$ingredients)
View(recipes[recipes[,"yuzu"]==TRUE,])
recipes$egg <- grepl("egg", recipes$ingredients)
View(recipes[recipes[,"egg"]==TRUE,])
recipes$bacon <- grepl("bacon",recipes$ingredients)
View(recipes[recipes[,"bacon"]==TRUE,])
recipes$tomato <- grepl("tomato",recipes$ingredients)
View(recipes[recipes[,"tomato"]==TRUE,])
recipes$fish_sauce <- grepl("fish sauce",recipes$ingredients)
View(recipes[recipes[,"fish_sauce"]==TRUE,])
recipes$soy_sauce <- grepl("soy sauce",recipes$ingredients)
View(recipes[recipes[,"soy_sauce"]==TRUE,])
fit<-rpart(cuisine~ingredient_count+yuzu+egg+bacon+tomato+fish_sauce+soy_sauce, data=recipes, method="class",control=rpart.control(minsplit=30, cp=0.001))
fancyRpartPlot(fit)

