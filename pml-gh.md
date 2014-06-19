predmachlearn project
========================================================
# http://etaoinbe.github.io/predmachlearn-project/ 

First attempt at machine learning was with a decision tree approach. The first tree that came out depended only on X. 
However the predict function didnt like it. When I looked at X I saw it was like a sequence no with no real predicting value. 
In the end I removed several more variables amongst which the columns that are all NA. 

What tripped me up for quite some time is that read.csv converts incomplete
numeric fields to factors. This gave memory errors and general instability in R.
 
Then I finally got some results. However accuracy was really bad only like 40%. 
So I decided to go for random forest next. I first tried to run it on the full training set but this never ended. 
So in the end I reduced the set to 3000 records, this claimed 100% accuracy. 
Apparently that is close to the truth as the submission page accepted all the results. 


```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(rattle)
```

```
## Rattle: A free graphical interface for data mining with R.
## Version 3.0.2 r169 Copyright (c) 2006-2013 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
#http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf

#
testmethod="rf"
#testmethod="rpart" 

#####################
# TRAINING
#####################

setwd("C:\\data\\lectures\\predmachlearn\\project") 
#training <- read.csv("pml-training.csv")
#xx  <- read.csv("pml-training.csv",as.is=T,stringsAsFactors=F)

training <- read.csv("pml-training.csv",as.is=T,stringsAsFactors=F)
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
plot(table(training$classe))
```

![plot of chunk unnamed-chunk-1](figure/unnamed-chunk-1.png) 

```r
ns=names(training)
for(i in 1:length(ns)) { 
  name=ns[i] 
  if(  name!="classe") { 
    if( typeof(training[,name])=="character"  ) {  
                                                   training[, c(name)]=as.numeric( training[, c(name)] )  ;
    }
    #print(sprintf("types tr %s    name %s ",typeof(training[, c(name)]), name ) )
  }}
```

```
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
```

```r
training$classe=as.factor(training$classe)

#training<- training[ sample(dim(training)[1], 100), ] #!!!
```

See how the dataset is boobytrapped.


```r
qplot(seq_along(training$classe),training$classe)
```

![plot of chunk unnamed-chunk-2](figure/unnamed-chunk-2.png) 


```r
qplot(training$X,training$classe)
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3.png) 


```r
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
  plot(modfit$finalModel,uniform=TRUE,main="tree")
  text(modfit$finalModel,use.n=TRUE,all=TRUE,cex=.8)
  fancyRpartPlot(modfit$finalModel)  
  confusionMatrix(training$classe, predict(modfit, training))
}

### 
```


```r
if(testmethod=="rf") {  
#  training<- training[ sample(dim(training)[1], 100), ] #!!!
  modfitrf=train(training$classe ~ ., method="rf", data=training, prox=TRUE )
  confusionMatrix(training$classe, predict(modfitrf, training))
}
```

```
  Confusion Matrix and Statistics
  
            Reference
  Prediction   A   B   C   D   E
           A 820   0   0   0   0
           B   0 590   0   0   0
           C   0   0 521   0   0
           D   0   0   0 510   0
           E   0   0   0   0 559
  
  Overall Statistics
                                       
                 Accuracy : 1          
                   95% CI : (0.9988, 1)
      No Information Rate : 0.2733     
      P-Value [Acc > NIR] : < 2.2e-16  
                                       
                    Kappa : 1          
   Mcnemar's Test P-Value : NA         
  
  Statistics by Class:
  
                       Class: A Class: B Class: C Class: D Class: E
  Sensitivity            1.0000   1.0000   1.0000     1.00   1.0000
  Specificity            1.0000   1.0000   1.0000     1.00   1.0000
  Pos Pred Value         1.0000   1.0000   1.0000     1.00   1.0000
  Neg Pred Value         1.0000   1.0000   1.0000     1.00   1.0000
  Prevalence             0.2733   0.1967   0.1737     0.17   0.1863
  Detection Rate         0.2733   0.1967   0.1737     0.17   0.1863
  Detection Prevalence   0.2733   0.1967   0.1737     0.17   0.1863
  Balanced Accuracy      1.0000   1.0000   1.0000     1.00   1.0000
```


```r
#####################
# TESTING
#####################
testing <- read.csv("pml-testing.csv",as.is=T,stringsAsFactors=F)
ns=names(testing)
for(i in 1:length(ns)) { 
  name=ns[i] 
  if(  name!="classe") { 
    if( typeof(testing[,name])=="character"  ) { 
                                                  testing[, c(name)]=as.numeric( testing[, c(name)] )  ;
    }
#    print(sprintf("types tr %s    name %s ",typeof(testing[, c(name)]), name ) )
  }}
```

```
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
## Warning: NAs introduced by coercion
```

```r
#NAs2 <- apply(testing,2,function(x) {sum(is.na(x))}) 
testing <- testing[,which(NAs == 0)] 
#testing<-testing[,colSums(is.na(testing)) < nrow(testing) ] 

removeIndex <- grep(excludes,names(testing))
#testing <- subset( testing, select = removeIndex )
testing <- testing[,-removeIndex]
if(testmethod=="rpart") {
  predict(modfit, testing, verbose = TRUE)
}
```



```r
if(testmethod=="rf") {
  predict(modfitrf, testing, verbose = TRUE)
}
```

Number of training samples: 3000 
Number of test samples:     0 

rf : 20 unknown predictions were added

 [1] B A B A A E D B A A B C B A E E A B B B
Levels: A B C D E




