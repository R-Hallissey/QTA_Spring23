#############################
# Quantitative Text Analysis 
# Homework 2 
# Ruairí Hallissey             
# 1-3-2023 / 11 Ventôse CCXXXI 
###############################

## Load packages
install.packages("caret",type="binary")
library("caret")
install.packages("caret", type="binary")
install.packages("vctrs", version="0.5.1", lib = "C:/R/win-library/4.2")
install.packages("doParallel")

pkgTest <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[,  "Package"])]
  if (length(new.pkg)) 
    install.packages(new.pkg,  dependencies = TRUE)
  sapply(pkg,  require,  character.only = TRUE)
}

lapply(c("tidyverse",
         "guardianapi",
         "quanteda", 
         "lubridate",
         "quanteda.textmodels", 
         "quanteda.textstats", 
         "caret", # For train/test split
         "MLmetrics", # For ML
         "doParallel"), # For parallel processing
       pkgTest)
# Data from https://www.dropbox.com/sh/o9bhqm65tf1bnuv/AADjLpLGfjqA4WLmoANtVRK7a/_Homeworks/hw2/data?dl=0&subfolder_nav_tracking=1 

### Problem 1 ###
# 1. Creating Corpus
yelp_corpus <- corpus(yelp_data_small,
               meta = list(), # including meta data for names
               unique_docnames = TRUE)
summary(yelp_corpus)

table(yelp_corpus$sentiment)
4912 / length(yelp_corpus$sentiment)
print("Probaility of a positive review is 0.49")

# 2. Processing Tokens and Creating Document Feature Matrix

# Fuctions
prep_toks <- function(text_corpus){
  toks <- tokens(text_corpus,
                 include_docvars = TRUE) %>%
    tokens_tolower() %>%
    tokens_remove(stopwords("english"), padding = TRUE) %>%
    tokens_remove('[\\p{P}\\p{S}]', valuetype = 'regex', padding = TRUE)
  return(toks)
}

get_coll <-function(tokens){
  unsup_col <- textstat_collocations(tokens,
                                     method = "lambda",
                                     size = 2,
                                     min_count = 5,
                                     smoothing = 0.5)
  
  unsup_col <- unsup_col[order(-unsup_col$count),] # sort detected collocations by count (descending)
  return(unsup_col)
}

prepped_toks <- prep_toks(yelp_corpus) # basic token cleaning
collocations <- get_coll(prepped_toks) # get collocations
toks <- tokens_compound(prepped_toks, pattern = collocations[collocations$z > 10,])
toks <- tokens_remove(tokens(toks), "") 

toks <- tokens(toks, 
               remove_numbers = TRUE,
               remove_punct = TRUE,
               remove_symbols = TRUE,
               #remove_hyphens = TRUE,
               remove_separators = TRUE,
               remove_url = TRUE)

#lowercase
toks <- tokens_tolower(toks)

## Remove stop words
stop_list <- stopwords("english")
toks <- tokens_remove(toks, stop_list)

toks <- tokens_wordstem(toks)

# Create dfm
yelp_dfm <- dfm(toks) # create DFM
yelp_dfm <- dfm_trim(yelp_dfm, min_docfreq = 40) # trim DFM
yelp_dfm <- dfm_tfidf(yelp_dfm)

# Create data frame
yelp_df <- convert(yelp_dfm, to = "data.frame", docvars = NULL)
yelp_df <- yelp_df[, -1] # drop document id variable (first variable)
sentiment_labels <- yelp_dfm@docvars$sentiment # get sentiment labels
yelp_df <- as.data.frame(cbind(sentiment_labels, yelp_df)) # labelled data frame

## ML Preparation
# You need to a) Create a 5% validation split
set.seed(2023) # set seed for replicability
yelp_df <- yelp_df[sample(nrow(yelp_df)), ] # randomly order labelled dataset
split <- round(nrow(yelp_df) * 0.05) # determine cutoff point of 5% of documents
vdata <- yelp_df[1:split, ] # validation set
ldata <- yelp_df[(split + 1):nrow(yelp_df), ] #labelled dataset minus validation set

summary(ldata)
ldata$sentiment_labels

#b) Create an 80/20 test/train split
train_row_nums <- createDataPartition(ldata$sentiment_labels, 
                                      p=0.2, 
                                      list= FALSE) # set human_labels as the Y variable in caret
Train <- ldata[train_row_nums, ] # training set
Test <- ldata[-train_row_nums, ] # testing set

#             c) Create five-fold cross validation with 3 repeats object - to supply to train()
train_control <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs= TRUE, 
  summaryFunction = multiClassSummary,
  selectionFunction = "best", # select the model with the best performance metric
  verboseIter = TRUE
)

# Default paramter
modelLookup(model = "naive_bayes")

#
tuneGrid = expand.grid(laplace = c(0,0.5,1.0),
                       usekernel = c(TRUE, FALSE),
                       adjust=c(0.75, 1, 1.25, 1.5))




cl <- makePSOCKcluster(2) # create number of copies of R to run in parallel and communicate over sockets
# Note that the number of clusters depends on how many cores your machine has.  You may need to reduce it to 3 for example.
registerDoParallel(cl) # register parallel backed with foreach package

# train model
nb_train <- train(sentiment_labels ~ ., 
                  data = Train,  
                  method = "naive_bayes", 
                  metric = "F1",
                  trControl = train_control,
                  tuneGrid = expand.grid(laplace = c(0,1),
                                         usekernel = c(TRUE, FALSE),
                                         adjust = c(0.75, 1, 1.25, 1.5)),
                  allowParallel= TRUE
)

# 
stopCluster(cl) # stop parallel process once job is done
print(nb_train)

# Evaluate performance 
pred <- predict(nb_train, newdata = Test) # generate prediction on Test set using training set model
head(pred)


# confusion matrix
confusionMatrix(reference = as.factor(Test$sentiment_labels), data = pred, mode='everything', positive='neg')

#             i) Finalise the model
nb_final <- train(sentiment_labels ~ ., 
                  data = ldata,  
                  method = "naive_bayes", 
                  trControl = trainControl(method = "none"),
                  tuneGrid = data.frame(nb_train$bestTune))

#             j) Save the model!
saveRDS(nb_final, "ASDS/Quantitative Text Analysis/Howework 2/nb_final")

#             k) If your machine is running slow... read in the model 
#nb_final <- readRDS("data/nb_final")

#             l) Predict from validation set
pred2 <- predict(nb_final, newdata = vdata)
head(pred2) # first few predictions

#             m) Evaluate confusion matrix (because we actually have labels...)
confusionMatrix(reference = as.factor(vdata$sentiment_labels), data = pred2, mode='everything')

## 4. Training a Support Vector Machine
# You need to a) Examine parameters 
modelLookup(model = "svmLinear")

#             b) Create a grid
tuneGrid <- expand.grid(C = c(0.5, 1, 1.5))

#             c) Set up parallel processing
cl <- makePSOCKcluster(2) # using 6 clusters. 
registerDoParallel(cl)

#             d) Train the model
svm_train <- train(section_labels ~ ., 
                   data = Train,  
                   method = "svmLinear", 
                   metric = "F1",
                   trControl = train_control,
                   tuneGrid = tuneGrid,
                   allowParallel= TRUE
)

#             e) Save the model!
saveRDS(svm_train, "Homework 2/svm_train")

#             f) If your machine is running slow... read in the model
#svm_train <- readRDS("data/svm_train") 

#             g) Stop the cluster
stopCluster(cl)

#             h) Evaluate performance
print(svm_train)
pred_svm <- predict(svm_train, newdata = Test) # Predict on test sample using best model
confusionMatrix(reference = as.factor(Test$section_labels), data = pred_svm, mode='everything')

#             i) Finalise by training on all labelled data
svm_final <- train(section_labels ~ ., 
                   data = ldata,  
                   method = "svmLinear", 
                   trControl = trainControl(method = "none"),
                   tuneGrid = data.frame(svm_train$bestTune))
print(svm_final)

#             j) Save the model!
saveRDS(svm_final, "Homework 2/svm_final")

#             k) In case your computer is running slow... read in the model
#svm_final <- readRDS("data/svm_final")

#             l) Predict from validation set
svm_pred2 <- predict(svm_final, newdata = vdata)

#             m) Evaluate confusion matrix
confusionMatrix(reference = as.factor(vdata$section_labels), data = svm_pred2, mode='everything')
