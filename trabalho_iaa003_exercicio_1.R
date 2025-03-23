install.packages("e1071")
install.packages("randomForest")
install.packages("mlbench")
install.packages("caret")
library(mlbench)
library(caret)

# TAREFAS
# 1. Carregue a base de dados Satellite

data(Satellite)

# 2. Crie Partições contendo 80% para treino e 20% para teste

df <- Satellite[c(17:20, 37)]

set.seed(7)

indices <- createDataPartition(df$classes, p = 0.80, list = FALSE)

df.treino <- df[indices,]
df.teste <- df[-indices,]

# 3. Treine modelos RandomForest, SVM, e RNA para predição destes dados

#### RandonForest ####

rf <- train(classes~., data=df.treino, method="rf")
predicoes.rf <- predict(rf, df.teste)
predicoes.rf
cm.rf <- confusionMatrix(predicoes.rf, df.teste$classes)
cm.rf # Accuracy : 0.8395

#### SVM ####

svm <- train(classes~., data=df.treino, method="svmRadial")
predicoes.svm <- predict(svm, df.teste)
cm.svm <- confusionMatrix(predicoes.svm, df.teste$classes)
cm.svm # Accuracy : 0.8679

#### RNA ####

rna <- train(classes~., data=df.treino, method="nnet", trace=FALSE)
precicoes.rna <- predict(rna, df.teste)
cm.rna <- confusionMatrix(predicoes.rf, df.teste$classes)
cm.rna # Accuracy : 0.8395 

# 4. Escolha o melhor modelo com base em suas matrizes de confusão



# 5. Indique qual modelo dá o melhor resultado e a métrica utilizada

