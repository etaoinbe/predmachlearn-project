# What you should submit
# 
# The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable
# in the training set. You may use any of the other variables to predict with. You should create a report describing 
# how you built your model, how you used cross validation, what you think the expected out of sample error is, and why
# you made the choices you did. You will also use your prediction model to predict 20 different test cases. 
# 
# 1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing
# your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5.
# It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed 
# online (and you always want to make it easy on graders :-).
# 2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. 
# Please submit your predictions in appropriate format to the programming assignment for automated grading. 
# See the programming assignment for additional details. 
# 
# Reproducibility 
# 
# Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your
# classmates. Please be sure that if they download the repo, they will be able to view the compiled HTML version of your
# analysis. 
# C:\data\git\predmachlearn-project 

library(caret)
library(rattle)
#http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf

#
testmethod="rf"
#testmethod="rpart" 

#####################
# TRAINING
#####################

#setwd("C:\\data\\lectures\\predmachlearn\\project") 
setwd("e:\\data-e\\project") 

#training <- read.csv("pml-training.csv")
#xx  <- read.csv("pml-training.csv",as.is=T,stringsAsFactors=F)

training <- read.csv("pml-training.csv",as.is=T,stringsAsFactors=F)

ns=names(training)
for(i in 1:length(ns)) { 
  name=ns[i] 
  if(  name!="classe") { 
    if( typeof(training[,name])=="character"  ) { cat("!!!",name);
                                                   training[, c(name)]=as.numeric( training[, c(name)] )  ;
    }
    #print(sprintf("types tr %s    name %s ",typeof(training[, c(name)]), name ) )
  }}
training$classe=as.factor(training$classe)

#training<- training[ sample(dim(training)[1], 100), ] #!!!

qplot(seq_along(training$classe),training$classe)
qplot(training$X,training$classe)

excludes="timestamp|X|user_name|new_window"

#training<-training[,colSums(is.na(training)) < nrow(training) ] 
#testing<-testingsrc[,colSums(is.na(testingsrc)) < nrow(testingsrc) ]
#training1 <- subset( trainingsrc, select = -X )
NAs <- apply(training,2,function(x) {sum(is.na(x))}) 
training <- training[,which(NAs == 0)] 
removeIndex <- grep(excludes, names(training))
training <- training[,-removeIndex]


if(testmethod=="rpart") {
  set.seed(975)
  modfit=train(training$classe ~ ., method="rpart", data=training )
  print(modfit$finalModel)
  jpeg("modfittree.jpg")
  plot(modfit$finalModel,uniform=TRUE,main="tree")
  text(modfit$finalModel,use.n=TRUE,all=TRUE,cex=.8)
  dev.off()
  jpeg("fancytree.jpeg")
  fancyRpartPlot(modfit$finalModel)
  dev.off()
  
  confusionMatrix(training$classe, predict(modfit, training))
}

### 
if(testmethod=="rf") {  
#  training<- training[ sample(dim(training)[1], 100), ] #!!!
  training<- training[ sample(dim(training)[1], 3000), ] #!!!

  modfitrf=train(training$classe ~ ., method="rf", data=training, prox=TRUE )
  confusionMatrix(training$classe, predict(modfitrf, training))
}

#####################
# TESTING
#####################
testing <- read.csv("pml-testing.csv",as.is=T,stringsAsFactors=F)
ns=names(testing)
for(i in 1:length(ns)) { 
  name=ns[i] 
  if(  name!="classe") { 
    if( typeof(testing[,name])=="character"  ) { cat("!!!",name);
                                                  testing[, c(name)]=as.numeric( testing[, c(name)] )  ;
    }
    print(sprintf("types tr %s    name %s ",typeof(testing[, c(name)]), name ) )
  }}

#NAs2 <- apply(testing,2,function(x) {sum(is.na(x))}) 
testing <- testing[,which(NAs == 0)] 
#testing<-testing[,colSums(is.na(testing)) < nrow(testing) ] 

removeIndex <- grep(excludes,names(testing))
#testing <- subset( testing, select = removeIndex )
testing <- testing[,-removeIndex]
table(training$classe)
plot(table(training$classe))
if(testmethod=="rpart") {
  predict(modfit, testing, verbose = TRUE)
}
if(testmethod=="rf") {
  prediction= predict(modfitrf, testing, verbose = TRUE)
}



pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(prediction)



