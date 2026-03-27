# install packages
# Google: R torch

install.packages("torch")
install.packages("luz")
install.packages("torchdatasets")

install_torch()

library(torch)
library(luz)
library(tibble)

# tensor: matrix/vector/scalar/...
torch_tensor(1)

library(torchdatasets)

# Data

sd = 0.1
# training data
# training data will be used to train the model 
# data assumption: Y = 2 cos(X_1) + sin(X_2) + \epsilon
N_train = 10000
X_train_1 <- rnorm(N_train)
X_train_2 <- rnorm(N_train)
epsilon_train <- rnorm(N_train, sd = sd)
Y_train <- 2 * cos(X_train_1) + sin(X_train_2) + epsilon_train
# data_train <- matrix(c(X_train, Y_train), ncol = 2)

# validation data
# To check whether the model has been overfitted at some point
N_valid = 5000
X_valid_1 <- rnorm(N_valid)
X_valid_2 <- rnorm(N_valid)
epsilon_valid <- rnorm(N_valid, sd = sd)
Y_valid <- 2 * cos(X_valid_1) + sin(X_valid_2) + epsilon_valid
# data_valid <- matrix(c(X_valid, Y_valid), ncol = 2)

# test data
# test data will be used only for evaluation of the model after the 
# training is complete.
N_test = 5000
X_test_1 <- rnorm(N_test)
X_test_2 <- rnorm(N_test)
epsilon_test <- rnorm(N_test, sd = sd)
Y_test <- 2 * cos(X_test_1) + sin(X_test_2) + epsilon_test
# data_test <- matrix(c(X_test, Y_test), ncol = 2)

X_train <- matrix(c(X_train_1, X_train_2), ncol = 2)
X_valid <- matrix(c(X_valid_1, X_valid_2), ncol = 2)
X_test <- matrix(c(X_test_1, X_test_2), ncol = 2)

# After having data, transform them into a tensor
data_train_tensor_X <- torch_tensor(X_train)
data_valid_tensor_X <- torch_tensor(X_valid)
data_test_tensor_X <- torch_tensor(X_test)

data_train_tensor_Y <- torch_tensor(Y_train)
data_valid_tensor_Y <- torch_tensor(Y_valid)
data_test_tensor_Y <- torch_tensor(Y_test)

# After having the tensor, I will create a tensor-specific dataset

my_dataset <- dataset(
  name = "my_dataset",
  
  initialize = function(X, y) {
    self$X <- X
    self$y <- y
  },
  
  .getitem = function(i) {
    list(
      # X: matrix n * 1
      x = self$X[i, ],
      # Y: vector of n
      y = self$y[i]
    )
  },
  
  .length = function() {
    self$X$size()[1]
  }
)

data_train_tensor_dataset <- my_dataset(data_train_tensor_X, data_train_tensor_Y)
data_valid_tensor_dataset <- my_dataset(data_valid_tensor_X, data_valid_tensor_Y)
data_test_tensor_dataset <- my_dataset(data_test_tensor_X, data_test_tensor_Y)

# after having dataset, load them into the dataloader
# e.g. IF you have n =1000, batch_size = 10, then, you have 100 batches
train_dl <- dataloader(data_train_tensor_dataset, batch_size = 16, shuffle = T)
valid_dl <- dataloader(data_valid_tensor_dataset, batch_size = 16)
test_dl <- dataloader(data_test_tensor_dataset, batch_size = 16)

# Used for test
batch <- dataloader_make_iter(train_dl) %>% dataloader_next()

torch_manual_seed(777)

net <- nn_module(
  "corr-cnn",
  
  initialize = function() {
    
    # X -> (Z_1, Z_2, \ldots, Z_64)
    self$fc1 <- nn_linear(in_features = 2, out_features = 64)
    # (Z_1, Z_2, \ldots, Z_64) -> (W_1)
    self$fc2 <- nn_linear(in_features = 64, out_features = 1)
    self$fc3 <- nn_linear(in_features = 32, out_features = 16)
    self$fc4 <- nn_linear(in_features = 16, out_features = 1)
    
  },
  
  forward = function(x) {
    
    x %>% 
      self$fc1() %>%
      nnf_relu() %>%
      
      self$fc2() 
      # nnf_relu() %>%
      # 
      # self$fc3() %>% 
      # nnf_relu() %>%
      # 
      # self$fc4()
  }
)

# luz package: easy to run a model
fitted <- net %>%
  setup(
    loss = function(y_hat, y_true) nnf_mse_loss(y_hat, y_true$unsqueeze(2)),
    optimizer = optim_adam
  ) %>%
  fit(train_dl, epochs = 3, valid_data = valid_dl)

preds <- predict(fitted, test_dl)

preds <- preds$to(device = "cpu")$squeeze() %>% as.numeric()

df <- data.frame(preds = preds, targets = Y_test)

library(ggplot2)

ggplot(df, aes(x = targets, y = preds)) +
  geom_point(size = 0.1) +
  theme_classic() +
  xlab("true correlations") +
  ylab("model predictions")
