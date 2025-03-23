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

set.seed(10)

indices <- createDataPartition(df$classes, p = 0.80, list = FALSE)

df.treino <- df[indices,]
df.teste <- df[-indices,]

# 3. Treine modelos RandomForest, SVM, e RNA para predição destes dados

#### RandonForest ####

rf <- train(classes~., data=df.treino, method="rf")
predicoes.rf <- predict(rf, df.teste)
cm.rf <- confusionMatrix(predicoes.rf, df.teste$classes)

#### SVM ####

svm <- train(classes~., data=df.treino, method="svmRadial")
predicoes.svm <- predict(svm, df.teste)
cm.svm <- confusionMatrix(predicoes.svm, df.teste$classes)

#### RNA ####

rna <- train(classes~., data=df.treino, method="nnet", trace=FALSE)
predicoes.rna <- predict(rna, df.teste)
cm.rna <- confusionMatrix(predicoes.rna, df.teste$classes)

# 4. Escolha o melhor modelo com base em suas matrizes de confusão

cm.rf # Accuracy : 0.838
cm.svm # Accuracy : 0.852
cm.rna # Accuracy : 0.771

# 5. Indique qual modelo dá o melhor resultado e a métrica utilizada

# O melhor modelo é do SVM, que apresenta uma acurácia de 85%, 
# enquanto o Random Forest apresenta 83% e por último o RNA com 77%
