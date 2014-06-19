library(mlbench)
data(Sonar)

set.seed(1)
inTrain <- createDataPartition(Sonar$Class)
training <- Sonar[inTrain[[1]], ]
testing <- Sonar[-inTrain[[1]], ]

pp <- preProcess(training[,-ncol(Sonar)])
training2 <- predict(pp, training[,-ncol(Sonar)])
training2$Class <- training$Class
testing2 <- predict(pp, testing[,-ncol(Sonar)])
testing2$Class <- testing2$Class

tc <- trainControl("repeatedcv", 
                   number=10, 
                   repeats=10, 
                   classProbs=TRUE, 
                   savePred=T)
set.seed(2)
RF <-  train(Class~., data= training, 
             method="rf", 
             trControl=tc)
#normal trainingData
set.seed(2)
RF.CS <- train(Class~., data= training, 
               method="rf", 
               trControl=tc, 
               preProc=c("center", "scale")) 