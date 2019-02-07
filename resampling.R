
library("httr")
library("readxl")
library("dplyr")
library("caret")
library("rpart")
library("DMwR")
library("imbalance")
library("ROSE")
library("ggplot2")
library("pROC")
library("gridExtra")

###### GETTING THE DATASET
url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
GET(url, write_disk(tf <- tempfile(fileext = ".xls")))
df <- read_excel(tf, na = "NA", skip = 1)

###### FUNCTIONS
collect_roc <- function(model, data_train, data_test) {
  roc_obj_train <- roc(data_train$class, 
    predict(model, data_train, type = "prob")[, "Yes"],
    levels = c("Yes", "No"))
  
  roc_obj_test <- roc(data_test$class, 
    predict(model, data_test, type = "prob")[, "Yes"],
    levels = c("Yes", "No"))
  
  roc <- list(train = roc_obj_train, test = roc_obj_test)
  return(roc)
}

plot_rocs <- function(model){
  list_test <- model$test
  list_train <- model$train
  model_name <- model$model_name
  
  val_train <- round(as.vector(pROC::ci(model$train))[2] * 100, 1)
  val_test <- round(as.vector(pROC::ci(model$test))[2] * 100, 1)
  
  df_tmp <- data.frame(
    data = "train",
    FPR = (1 - list_train$specificities),
    TPR = list_train$sensitivities
  )
  
  df_tmp <- rbind(df_tmp,
    data.frame(
      data = "test",
      FPR = (1 - list_test$specificities),
      TPR = list_test$sensitivities
      )
    )
  
  title_text <- ifelse(model_name == "original", "Model with original data", paste0("Model with ", model_name, "-sampling"))
  subtitle_text <- paste0("ROC train: ", val_train, "%; ", " test: ", val_test, "%")
  p <- ggplot(df_tmp, aes(FPR, TPR, colour = data)) + 
    geom_line(size = 1) + 
    labs(title = title_text, subtitle = subtitle_text, x = "FPR (1-Specificity)", y = "TPR (Sensitivity)")
  return(p)
}

plot_distributions <- function(sampling){
  df_name <- ifelse(sampling == "none", "train", paste0("train_", sampling))
  title_text <- ifelse(sampling == "none", "Original data", paste0("Data after ", sampling, "-sampling"))
  data <- get(df_name)
  p <- ggplot(data, aes(limit_bal, bill_amt1, colour = class)) +
    geom_point(alpha = 0.5) +
    labs(title = title_text, x = "Limit balance", y = "Bill AMT 1")
  return(p)
}

summarise_results <- function(results){
  res <- data.frame()
  for(ii in 1:length(results)){
    ROC_train <- as.vector(pROC::ci(results[[ii]]$train))
    ROC_test <- as.vector(pROC::ci(results[[ii]]$test))
    nn <- results[[ii]]$model_name
    tmp <- data.frame(t(data.frame(ROC_train, ROC_test)))
    tmp <- cbind(model = paste0(nn, c(" train", " test")), tmp)
    row.names(tmp) <- c()
    res <- rbind(res, tmp)
  }
  colnames(res) <- c("model", "lower", "ROC", "upper")
  return(res)
}

###### MINIMAL DATA PREPROCESSING

df <- data.frame(df)
names(df) <- tolower(names(df))

# SAVING AND REMOVING CUSTOMERS' IDS
ids <- df$id
df <- df %>% select(-id)

# RENAMING LAST COLUMN (DEPENDENT VARIABLE)
names(df)[ncol(df)] <- "class"

# CONVERTING COLUMNS TO FACTORS
cols <- c("sex", "education", "class","marriage", "pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6")
for(col in names(df)){
  if(col %in% cols){
    df[, col] <- as.factor(df[, col])
  }
}

# LEVELS FOR CLASS VARIABLE
levels(df$class) <- c("No", "Yes")

# DATA PARTITIONING
set.seed(42)
partition <- createDataPartition(df$class, p = 0.7, list = F)
train <- df[partition, ]
test <- df[-partition, ]

###### MODELS WITH DIFFERENT RESAMPLING TECHNIQUES

# Original data
table(train$class) / nrow(train) * 100
model_none <- rpart(class ~ ., data = train, control = rpart.control(cp = 0.05))

# DOWNsampled data
train_down <- downSample(train[, 1:ncol(train)-1], train$class, yname = "class")
table(train_down$class) / nrow(train_down) * 100
model_down <- rpart(class ~ ., data = train_down, control = rpart.control(cp = 0.05))

# UPsampled data
train_up <- upSample(train[, 1:ncol(train)-1], train$class, yname = "class")
table(train_up$class) / nrow(train_up) * 100
model_up <- rpart(class ~ ., data = train_up, control = rpart.control(cp = 0.05))

# ROSEd data
train_rose <- ROSE(class ~ ., data = train, N = 32710, p=0.5, seed = 42)
train_rose <- train_rose$data
table(train_rose$class) / nrow(train_rose) * 100
model_rose <- rpart(class ~ ., data = train_rose, control = rpart.control(cp = 0.05))

# SMOTEd data
train_smote <- SMOTE(class ~., data = train)
table(train_smote$class) / nrow(train_smote) * 100
model_smote <- rpart(class ~ ., data = train_smote, control = rpart.control(cp = 0.05))

# MWMOTEd data
train_mwmote <- mwmote(train, numInstances = 10000, classAttr = "class")
train_mwmote <- rbind(train, train_mwmote)
table(train_mwmote$class) / nrow(train_mwmote) * 100
model_mwmote <- rpart(class ~ ., data = train_mwmote, control = rpart.control(cp = 0.05))

###### EVALUATING OBTAINED MODELS 
samplings <- c("none",  "down", "up", "rose", "smote", "mwmote")
models <- list(original = model_none,  down = model_down, up = model_up, 
  rose = model_rose, smote = model_smote, mwmote = model_mwmote
)

# COLLECTING RESULTS
results <- lapply(models, collect_roc, data_train = train, data_test = test)

# ADDING SAMPLING NAME TO RESULTS LIST
for(name in names(models)){
  results[[name]]$model_name <- name
}

# GENERATING PLOTS
distributions <- lapply(samplings, plot_distributions)
do.call(grid.arrange, distributions)

rocs <- lapply(results, plot_rocs)
do.call(grid.arrange, rocs)

# SUMMARIZING RESULTS IN TABLE
tbl <- summarise_results(results)
tbl
