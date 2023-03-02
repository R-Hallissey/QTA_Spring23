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
prep_toks <- function(yelp_corpus){
  toks <- tokens(yelp_corpus,
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
# Create dfm
yelp_dfm <- dfm(toks) # create DFM
yelp_dfm <- dfm_trim(yelp_dfm, min_docfreq = 10) # trim DFM
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
                                      p=0.8, 
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

#

cl <- makePSOCKcluster(6) # create number of copies of R to run in parallel and communicate over sockets
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
confusionMatrix(reference = Test$human_labels, data = pred, mode='everything', positive='neg')