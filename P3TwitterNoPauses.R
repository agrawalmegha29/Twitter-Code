##
## Authors: Megha Agrawal
##
##

##
## Include all libraries to be used
##
library(RCurl)
library(bitops)
library(rjson)
library(streamR)
library(ROAuth)
library(tm)
library(stringr)
library(SnowballC)
library(RMOAjars)
library(rJava)
library(RMOA)

##
## Function to do pre-processing on the data (Tweet text) before feeding it to
##  the model for training or testing
##
preprocessData = function (tweets.df)
{
  # Remove all non-printable control characters
  usableText = str_replace_all(tweets.df$text,"[^[:graph:]]", " ")
  
  # Remove all hyperlinks
  usableText = str_replace_all(usableText, "http[s]*:[^ ]+", "")
  
  doc.vec <- VectorSource(usableText)
  doc.corpus <- Corpus(doc.vec)
  
  # Do all text transformations and cleanup
  # - Conversion to lower-case
  # - Stemming
  # - Removing punctuation, numbers, stop words, etc.
  doc.corpus <- tm_map(doc.corpus, tolower)
  doc.corpus <- tm_map(doc.corpus, stemDocument)
  doc.corpus <- tm_map(doc.corpus, removePunctuation)
  doc.corpus <- tm_map(doc.corpus, removeNumbers)
  
  doc.corpus <- tm_map(doc.corpus, stemDocument)
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("english"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("german"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("SMART"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("french"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("russian"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("spanish"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("swedish"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("portuguese"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("norwegian"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("italian"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("hungarian"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("finnish"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("dutch"))
  doc.corpus <- tm_map(doc.corpus, removeWords, stopwords("danish"))
  
  doc.corpus <- tm_map(doc.corpus, stripWhitespace)
  doc.corpus <- tm_map(doc.corpus, PlainTextDocument)
  
  TDM <- TermDocumentMatrix(doc.corpus)
  
  return (TDM)  
}

##
## Function to do some further processing on the training
## data, including creating a data-frame from the term-document
## matrix
##
processTrainData = function (TDM)
{
  # Transpose, so rows correspond to each record
  transposed = t(TDM)
  matrix_tdm = as.matrix(transposed)
  matrix_tdm = as.data.frame(matrix_tdm)
  
  love_records = matrix_tdm[matrix_tdm$love!= 0,]
  hate_records = matrix_tdm[matrix_tdm$hate!= 0,]
  love_records$class_features = "love"
  hate_records$class_features = "hate"
  
  final_data = rbind(love_records,hate_records)
  #final_data=sample(final_data)
  
  class_data= final_data[,"class_features"]
  drops = c("love","hate","class_features")
  final_data = final_data[,!(names(final_data) %in% drops)]
  
  final_data$row_sums = rowSums(final_data, dims=1)
  final_data$class_features = class_data
  final_data = final_data[final_data$row_sums != 0,]
  final_data$row_sums=NULL
  
  final_data[, 'class_features'] <- as.factor(final_data[, 'class_features'])
  return (final_data)
}

##
## Twitter connection and auth
##

# Twitter API keys
requestURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"
consumerKey <- "xxxxxyyyyyyyzzzzzzzzzzz"
consumerSecret <- "xxxxxyyyyyyyzzzzzzzzzzzxxxxxyyyyyyyzzzzzzzzzzz"

# Create the OAuth object with the given credentials
my_oauth <- OAuthFactory$new(consumerKey=consumerKey,
                             consumerSecret=consumerSecret, requestURL=requestURL,
                             accessURL=accessURL, authURL=authURL)

# Perform the OAuth handshake,  At this point there should be a
#  prompt to enter the code provided by Twitter after the command above
my_oauth$handshake(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl"))



##
## Model + Training
##

# Get the tweets for generating the training data set
# The tweets are for the track words 'love' and 'hate' and we keep
#  receiving the tweets for 'timeout' number  of seconds
if(file.exists("tweets.json") == TRUE) {
  file.remove("tweets.json")
}

filterStream("tweets.json", track = c("love"), timeout = 50, oauth = my_oauth)
filterStream("tweets.json", track = c("hate"), timeout = 50, oauth = my_oauth)
# Read the received tweets into a data frame
tweets.df <- parseTweets("tweets.json", simplify = TRUE)

# Pre-process and prepare training data for making the model
TDM = preprocessData(tweets.df)

freq_words = findFreqTerms(TDM,lowfreq= 80)

print(freq_words)
TDM = TDM[freq_words,]
#inspect(TDM[1:10,])

final_data = processTrainData(TDM)
final_data= final_data[sample(nrow(final_data)),]

model_vars <- setdiff(colnames(final_data), list('class_features'))

# Create the formula for the model
model_formula <- as.formula(paste('class_features',paste(model_vars, collapse=' + '),sep=' ~ '))

print(model_formula)

# Initialize an untrained MOA model based on the HoeffdingTree
hdt <- HoeffdingTree(numericEstimator = "GaussianNumericAttributeClassObserver")
final_data[, 'class_features'] <- as.factor(final_data[, 'class_features'])

final_data <- factorise(final_data)

# Now prepare a streaming data frame from the training data
final_data_datastream <- datastream_dataframe(data=final_data)

# Train the Hoeffding Tree model with the training dataset
mymodel <- trainMOA(model = hdt,
                    formula = model_formula,
                    data = final_data_datastream)

print(mymodel)

###
### Continuous Testing + Updation
###

while (TRUE)
{
  # Get the tweets that will be used for generating the test data
  if(file.exists("tweets_test.json") == TRUE) {
    file.remove("tweets_test.json")
  }
  filterStream("tweets_test.json", track = c("love"), timeout = 10, oauth = my_oauth)
  filterStream("tweets_test.json", track = c("hate"), timeout = 10, oauth = my_oauth)
  # Read the received tweets into a data frame
  twitterTest.df <- parseTweets("tweets_test.json", simplify = TRUE)
  
  # Pre-process the test data before preparing for prediction
  TDMTest = preprocessData(twitterTest.df)
  
  # Transpose, so rows correspond to each record
  TDM_transposedTest = t(TDMTest)
  Matrix_Test = as.matrix(TDM_transposedTest)
  DataFrame_Test = as.data.frame(Matrix_Test)
  
  # Separate the love records
  love_records = DataFrame_Test[DataFrame_Test$love!= 0,]
  # Separate the hate records
  hate_records = DataFrame_Test[DataFrame_Test$hate!= 0,]
  
  # Add the classes for later comparison
  love_records$class_features = "love"
  hate_records$class_features = "hate"
  
  # Combine back the records
  DataFrame_Test = rbind(love_records, hate_records)
  
  # Shuffle the rows so that the distribution of love and hate records
  #  in the dataset is random
  DataFrame_Test= DataFrame_Test[sample(nrow(DataFrame_Test)),]
  
  # Now get the classes in a vector, and nullify them
  #  before feeding the test set to the model
  test_data_class =  DataFrame_Test$class_features;
  DataFrame_Test$class_features = NULL;
  
  # Remove the 'love' and 'hate' columns
  drops = c("love","hate")
  DataFrame_Test = DataFrame_Test[, !(names(DataFrame_Test) %in% drops)]
  
  # Keep only those columns in the test data set that are also in the training set
  DataFrame_Test = DataFrame_Test[,intersect(colnames(final_data), colnames(DataFrame_Test))]
  
  # In case testing set doesn't have some columns that the training set had, add them
  #  and set them to 0
  diff_cols <- setdiff(colnames(final_data), c(colnames(DataFrame_Test),'class_features'))
  DataFrame_Test[,diff_cols]=0
  
  ##
  ## Predict using the HoeffdingTree model
  ##
  scores <- predict(mymodel, newdata=DataFrame_Test, type="response")
  confusion_matrix = table(scores, test_data_class)
  
  print("\n Confusion Matrix: \n")
  print(confusion_matrix)
  # Pray, and print the results
  print("\n Accuracy of the model: ")
  if(NROW(confusion_matrix) == 2)
  {
    TP = confusion_matrix["love","love"]
    TN = confusion_matrix["hate","hate"]
    FP = confusion_matrix["love","hate"]
    FN = confusion_matrix["hate","love"]
    
  }
  else
  {
    TP = confusion_matrix["love","love"]
    FP = confusion_matrix["love","hate"]
    TN = 0
    FN = 0
  }
  accuracy = (TP+TN)/(TP+TN+FP+FN) 
  print(accuracy)
  
  #final_data_feature = final_data$class_features
  #final_data$class_features = NULL
  #scores <- predict(mymodel, newdata=final_data, type="response")
  #table(scores, final_data_feature)
  
  # Updating the Hoeffding Tree model with the testing dataset
  DataFrame_Test$class_features = test_data_class
  DataFrame_Test <- factorise(DataFrame_Test)
  test_data_datastream <- datastream_dataframe(data=DataFrame_Test)
  
  mymodel <- trainMOA(model = hdt,
                      formula = model_formula,
                      reset = FALSE,
                      data = test_data_datastream)
}
