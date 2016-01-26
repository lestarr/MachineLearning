#Machine Learning

#Goal: predict the manner of doing an excercize on several measurement
setwd("C:\\2Projects\\DataAnalysis\\Rprogs\\wdir")
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url = fileUrl, destfile = "./data/pml-training.csv")
# download test data
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url = fileUrl, destfile = "./data/pml-testing.csv")





library(YaleToolkit)
library(dplyr)
library(caret)
setwd("C:\\2Projects\\DataAnalysis\\Rprogs\\wdir")

#load data
train <- read.csv("data/pml-training.csv")


#exploratory - look at the predictors, eliminate sparse predictors
# unique() of whatis() shows only 2 values - 0 and 19216, so we eliminate every predictor, which has more than 0 missing values.
whatis_train <- whatis(train)
missing <- which(whatis_train$missing > 0)

# split into train and test
set.seed(1567)
inTrain <- createDataPartition(train$classe, p = 0.7, list = FALSE)
train_main <- train[inTrain,]
test_main <- train[-inTrain,]
train_clean <- train_main[,-missing]

nums <- sapply(train_clean,is.numeric) #find numerics
classe <- train_clean$classe # save output factor variable
train_clean_nums <- train_clean[,nums] # 
train_clean_nums <- cbind(train_clean_nums, classe)

#small train with all classes
strain <- train_clean_nums[c(1:100, 3907:4007, 6565:6665, 8961:9061, 11213:11313),]

#first rf
# prepare parallel processing
library(parallel)
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

#first 4 predictors lead to overfitting, remove them
train_last <- train_clean_nums[,c(-1,-2,-3,-4)]

fitControl_parCV2 <- trainControl(allowParallel = TRUE, method = "cv", number = 2)
system.time(rf<- train(classe~., method = "rf", data = train_last,  prox=TRUE, trControl = fitControl_parCV2))

stopCluster(cluster)
 