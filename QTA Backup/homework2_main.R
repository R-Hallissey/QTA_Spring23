#############################
# Quantitative Text Analysis 
# Homework 2 
# Ruairí Hallissey             
# 1-3-2023 / 11 Ventôse CCXXXI 
###############################

## Load packages
## Load packages
install.packages("caret",type="binary")
library("caret")
install.packages("vctrs", version="0.5.1", lib = "C:/R/win-library/4.2")
library(vctrs)
install.packages("naivebayes")
library(naivebayes)
install.packages("wordcloud")
library(wordcloud)

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
yelp_corpus <- corpus(yelp_data,
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
yelp_dfm <- dfm_trim(yelp_dfm, min_docfreq = 40) # drastically increased frequency cause when i ran at it took over 24hrs
                                                 # and by the time i stopped it i didn't have much time left
yelp_dfm <- dfm_tfidf(yelp_dfm)

# Create data frame
yelp_df <- convert(yelp_dfm, to = "data.frame", docvars = NULL)
yelp_df <- yelp_df[, -1] # drop document id variable (first variable)
sentiment_labels <- yelp_dfm@docvars$sentiment # get sentiment labels
yelp_df <- as.data.frame(cbind(sentiment_labels, yelp_df)) # labelled data frame

## ML Preparation
# Create a 5% validation split
set.seed(2023) # set seed for replicability
yelp_df <- yelp_df[sample(nrow(yelp_df)), ] # randomly order labelled dataset
split <- round(nrow(yelp_df) * 0.05) # determine cutoff point of 5% of documents
vdata <- yelp_df[1:split, ] # validation set
ldata <- yelp_df[(split + 1):nrow(yelp_df), ] #labelled dataset minus validation set

summary(ldata)

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
saveRDS(nb_final, "QTA/nb_final")

#             k) If your machine is running slow... read in the model 
#nb_final <- readRDS("data/nb_final")

#             l) Predict from validation set
pred2 <- predict(nb_final, newdata = vdata)
head(pred2) # first few predictions

#             m) Evaluate confusion matrix (because we actually have labels...)
confusionMatrix(reference = as.factor(vdata$sentiment_labels), data = pred2, mode='everything')

#Analysis
print("In the testing set accuracy is 0.726 (95% CI :0.6846 - 0.7647), meaning 72% of the true negatives and true positives are classified accurately. Classification accuracy is significant (p <.001) meaning there is less .001% chance of obtaining these results by chance.

Precision, being true positives divided by false and negative positives is .82, meaning 82% of the positives are classified correctly. 

Recall, true positives divided by false negative and true positive, is 0.5743,meaning is 57.43% of data is correctly identified as belonging to the positive class. 

F1 is 0.67 meaning, an average 67% of all possible categories are correctly classified.
")

#Justification
print("Minimum frequency in the DFM was drastically increased to 40. >10 term frequncy was running for over 24hrs, after which I pulled the plug, so I needed something that run quickly.

Examining performance indicators of both models, they are very similar, suggesting model is not overfit to the training data.

5% cross validation split was also created to test out of sample. Split data gave predictive terms similar to training and main model such as hard, compar,  place, like, and  well

Used tuning grid divide the domain of the hyperparameters then calculated  performance metrics using cross-validation. ")

#validation
nb_final$coefnames

kwic(toks, phrase("hard"))
print("Hard often appears in context of people describing found too tough")

kwic(toks, phrase("place"))
print("Place often appears as people compliant the resteraunt")

kwic(toks, phrase("well"))
print("Well often appears as people compliant the resteraunt")


## Support Vector Machine
# You need to a) Examine parameters 
modelLookup(model = "svmLinear")

#             b) Create a grid
tuneGrid <- expand.grid(C = c(0.5, 1, 1.5))

#             c) Set up parallel processing
cl <- makePSOCKcluster(6) # using 6 clusters. 
registerDoParallel(cl)

#             d) Train the model
svm_train <- train(sentiment_labels ~ ., 
                   data = Train,  
                   method = "svmLinear", 
                   metric = "F1",
                   trControl = train_control,
                   tuneGrid = tuneGrid,
                   allowParallel= TRUE
)

#             e) Save the model!
saveRDS(svm_train, "QTA/svm_train")

#             f) If your machine is running slow... read in the model
#svm_train <- readRDS("data/svm_train") 

#             g) Stop the cluster
stopCluster(cl)

#             h) Evaluate performance
print(svm_train)
pred_svm <- predict(svm_train, newdata = Test) # Predict on test sample using best model
confusionMatrix(reference = as.factor(Test$sentiment_labels), data = pred_svm, mode='everything')

#             i) Finalise by training on all labelled data
svm_final <- train(sentiment_labels ~ ., 
                   data = ldata,  
                   method = "svmLinear", 
                   trControl = trainControl(method = "none"),
                   tuneGrid = data.frame(svm_train$bestTune))
print(svm_final)
svm_final$coefnames

#             j) Save the model!
saveRDS(svm_final, "QTA/svm_final")

#             k) In case your computer is running slow... read in the model
#svm_final <- readRDS("data/svm_final")

#             l) Predict from validation set
svm_pred2 <- predict(svm_final, newdata = vdata)

#             m) Evaluate confusion matrix
confusionMatrix(reference = as.factor(vdata$sentiment_labels), data = svm_pred2, mode='everything')

# Analysis
print("In the testing set accuracy is0.834 (95% CI :0.7984 / 0.8656), meaning 82.31% of the true negatives and true positives are classified accurately. Classification accuracy is significant (p <.001) meaning there is less .001% chance of obtaining these results by chance.

Precision, being true positives divided by false and negative positives is 0.8487 , meaning 84.87% of the positives are classified correctly. 

Recall, true positives divided by false negative and true positive, is  0.8112 ,meaning is 81.12% of data is correctly identified as belonging to the positive class. 

F1 is  0.8296 meaning, an average 82.96% of all possible categories are correctly classified.

Super Vector Machine model has substnationally higher recall, and thus overall higher F1.
")

#Justication
print("Examining performance indicators of both training and test models, they are very similar, suggesting the model is not overfit to the training data.

SVM training and final models also gave predictive terms similar to thwe cross validation split.  

Used tuning grid divide the domain of the hyperparameters then calculated performance metrics using cross-validation") 


#Validation 
svm_final$coefnames

kwic(toks, phrase("hard"))
print("Hard often appears in context of people describing found too tough")

kwic(toks, phrase("place"))
print("Place often appears as people compliant the resteraunt")

kwic(toks, phrase("well"))
print("Well often appears as people compliant the resteraunt")

#################################################
# Problem 2
###############################################
#Data

dat$content <- str_replace(dat$content, "\u2022.+$", "")

# Creating corpus
corp <- corpus(dat, 
               docid_field = "title",
               text_field = "content")

# Processing
prepped_toks <- prep_toks(corp) # basic token cleaning
collocations <- get_coll(prepped_toks) # get collocations
toks <- tokens_compound(prepped_toks, pattern = collocations[collocations$z > 10,]) # replace collocations
toks <- tokens_remove(tokens(toks), "")  # let's also remove the whitespace placeholders
toks <- tokens_remove(toks, c("said","say"))

#Stopwords 
stop_list <- stopwords("english")
toks <- tokens_remove(toks, stop_list)


toks <- tokens(toks, 
               remove_numbers = TRUE,
               remove_punct = TRUE,
               remove_symbols = TRUE,
               #remove_hyphens = TRUE,
               remove_separators = TRUE,
               remove_url = TRUE)

# Minimum frequency = 20
dfm <- dfm(toks)
dfm <- dfm_trim(dfm, min_docfreq = 20)

# date fix
docvars(dfm, "date") <- lubridate::dmy(dfm@docvars$date)

#
stmdfm <- convert(dfm, to = "stm")

# Set k
K <- 35

# Run STM algorithm
#i struggled to get prevalence code working. I reconfigured date
# with lubridate above but did not have time to run the alogrithim again

modelFit <- stm(documents = stmdfm$documents,
                vocab = stmdfm$vocab,
                K = K,
                #prevalence = ~ s(month(date)),
                #prevalence = ~ s(as.numeric(date_month)), 
                data = stmdfm$meta,
                max.em.its = 500,
                init.type = "Spectral",
                seed = 1234,
                verbose = TRUE)


# Save your model!
saveRDS(modelFit, "QTA/modelFit")

# Load model (in case your computer is running slow...)
modelFit <- readRDS("data/modelFit")

## 3. Interpret Topic model 
# Inspect most probable terms in each topic
labelTopics(modelFit)

# plotting frequent terms
plot.STM(modelFit, 
         type = "summary", 
         labeltype = "frex", # plot according to FREX metric
         text.cex = 0.7,
         main = "Topic prevalence and top terms")

# Use wordcloud to visualise top terms per topic
cloud(modelFit,
      topic = 1,
      scale = c(2.5, 0.3),
      max.words = 50)

# Reading documents with high probability topics: the findThoughts() function
findThoughts(modelFit,
             texts = dfm@docvars$standfirst, # If you include the original corpus text, we could refer to this here
             topics = 1,
             n = 10)

######################################
#Couldn't process without correct date


## 4. Topic validation: predictive validity using time series data
#     a) Convert metadata to correct format
#stmdfm$meta$num_month <- as.numeric(stmdfm$meta$date_month)
stmdfm$meta$num_month <- month(stmdfm$meta$date)

#     b) Aggregate topic probability by month
agg_theta <- setNames(aggregate(modelFit$theta,
                                by = list(month = stmdfm$meta$num_month),
                                FUN = mean),
                      c("month", paste("Topic",1:K)))
agg_theta <- pivot_longer(agg_theta, cols = starts_with("T"))

#     c) Plot aggregated theta over time
ggplot(data = agg_theta,
       aes(x = month, y = value, group = name)) +
  geom_smooth(aes(colour = name), se = FALSE) +
  labs(title = "Topic prevalence",
       x = "Month",
       y = "Average monthly topic probability") + 
  theme_minimal()

## 5. Semantic validation (topic correlations)
topic_correlations <- topicCorr(modelFit)
plot.topicCorr(topic_correlations,
               vlabels = seq(1:ncol(modelFit$theta)), # we could change this to a vector of meaningful labels
               vertex.color = "white",
               main = "Topic correlations")

## 6. Topic quality (semantic coherence and exclusivity)
topicQuality(model = modelFit,
             documents = stmdfm$documents,
             xlab = "Semantic Coherence",
             ylab = "Exclusivity",
             labels = 1:ncol(modelFit$theta),
             M = 15)

# An alternative approach, using underlying functions
SemEx <- as.data.frame(cbind(c(1:ncol(modelFit$theta)), 
                             exclusivity(modelFit),
                             semanticCoherence(model = modelFit,
                                               documents = stmdfm$documents,
                                               M = 15)))

colnames(SemEx) <- c("k", "ex", "coh")

SemExPlot <- ggplot(SemEx, aes(coh, ex)) +
  geom_text(aes(label=k)) +
  labs(x = "Semantic Coherence",
       y = "Exclusivity",
       title = "Topic Semantic Coherence vs. Exclusivity") +
  geom_rug() +
  theme_minimal() +
  theme(panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_rect(colour = "gray", linewidth=1))
SemExPlot

# Inspect outliers
labelTopics(modelFit,
            topics = c(6,3))

## 7. Extended visualisations
#     a) Using LDAvis
toLDAvis(mod = modelFit,
         docs = stmdfm$documents,
         open.browser = interactive(),
         reorder.topics = TRUE)

#     b) Using stmBrowser
#        Warning: this will (silently) change your working directory!
stmBrowser(mod = modelFit,
           data = stmdfm$meta,
           covariates = c("section_name", "num_month"),
           text = "standfirst",
           n = 1000)

## 8. Estimating covariate effects
#     a) Calculate
estprop <- estimateEffect(formula = c(1:ncol(modelFit$theta)) ~ section_name + s(num_month),
                          modelFit,
                          metadata = stmdfm$meta,
                          uncertainty = "Global",
                          nsims = 25)

summary(estprop)

#     b) Plot topic probability differences
custom_labels <- seq(1:K)
plot.estimateEffect(x = estprop,
                    #model = modelFit,
                    method = "difference",
                    covariate = "section_name",
                    cov.value1 = "World",
                    cov.value2 = "Opinion",
                    topics = estprop$topics,
                    #xlim = c(-.05, .05),
                    labeltype = "custom",
                    custom.labels = custom_labels)

#     c) Plot topic probability over time (similar to above plot in 4.)
plot.estimateEffect(x = estprop,
                    #model = modelFit,
                    method = "continuous",
                    covariate = "num_month",
                    topics = estprop$topics,
                    #xlim = c(-.05, .05),
                    labeltype = "custom",
                    custom.labels = custom_labels,
                    printlegend = F) # Toggle this to see January

## 9. Using data to determine k
?searchK
kResult <- searchK(documents = stmdfm$documents,
                   vocab = stmdfm$vocab,
                   K=c(4:10),
                   init.type = "Spectral",
                   data = stmdfm$meta,
                   prevalence = ~ section_name + s(month(date)))
#cores = 6) # This no longer works on windows 10 :(
