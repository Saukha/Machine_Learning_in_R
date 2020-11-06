# DSSA-5201-Machine_Learning_Fundamentals # Assignment 2
# Write a linear regression algorithm from scratch, which prints coefficients and accuracy metrics 
# (e.g. R2) and plots the actual versus predicted response variables. 
# Compare the R2 of the results using your algorithm to the R2 of the results using lm() and predict().
# Document each step of the code to demonstrate you understand what each line of code does. The code 
# has to describe the steps of the linear regression model. For your algorithm, you may use basic 
# statistical functions and graphing functions but NOT machine learning functions (i.e. no lm(), no 
# predict() and related functions). Note, you must explain what solve() is doing if you use it (i.e. 
# an explanation for a general audience more than The function solves the equation).

# Given a matrix of observations X and the target Y. 
# The goal of the linear regression is to minimize the L2 norm 
# between Y and a linear estimate of Y :  Y_hat=WX. 
# Hence, linear regression can be rewritten as an optimization problem: 
# arg min_W ||Y-WX||^2. 
# A closed-form solution can easily be derived and the optimal W is
# W_hat = (X^T X)^{-1}X^T Y

library(ggplot2)
# Linear regression, using a closed-form solution
# define my_lm function, with the 
# init method where the model is fitted ----
my_lm <- function(x, y, intercept = TRUE, lambda = 0)
{
  # transform data to matrix if required
  if (!is.matrix(x))
  {
    x=as.matrix(x)
  }
  if (!is.matrix(y))
  {
    y=as.matrix(y)
  }
  # Add the intercept coefficient, if intercept is set to true
  if (intercept) 
  {
    x=cbind(1,x)
  }
  
  # create my_lm object ----
  my_lm = list(intercept = intercept)
  # Compute coefficients estimates
  # solve(t(x) %*% x) is used to invert the matrix while %*% denotes matrix multiplication, 
  # t(x) is the transpose of x and multiplying x results in a square matrix
  # solve(t(x) %*% x) invert it 
  my_lm[['coeffs']] = solve(t(x) %*% x) %*% t(x) %*% y
  # compute y_hat estimates for the train dataset
  my_lm[['preds']] = x %*% my_lm[['coeffs']]
  # compute the residuals
  my_lm[['residuals']] = my_lm[['preds']] - y
  # compute mean square error
  my_lm[['mse']] = mean(my_lm[['residuals']]^2)
  # compute root mean square error
  my_lm[['rmse']] = sqrt(my_lm[['mse']])
  # compute R^2
  my_lm[['rss']] = sum(my_lm[['residuals']]^2)  # residual sum of squares
  my_lm[['tss']] = sum((y - mean(my_lm[['preds']]))^2) # total sum of squares
  my_lm[['R2']] <- 1 -  my_lm[['rss']]/my_lm[['tss']]  # R sqaured
  # set the class of the my_lm object as "my_lm"
  attr(my_lm, "class") <- "my_lm"  
  return(my_lm)
}

# implement  
# predict method for the my_lm class ----
predict.my_lm <- function(my_lm, x,..)
{
  # transform data to matrix if required
  if (!is.matrix(x))
  {
    x = as.matrix(x)
  }
  # Add the intercept coefficient, if intercept is set to true  
  if (my_lm[["intercept"]])
  {
    x = cbind(1,x)
  }
  x %*% my_lm[["coeffs"]]
}

# # plot method for the my_lm class ----
# plot_residuals.my_lm <- function(my_lm, bins=20)
# {
#   library(ggplot2)
#   qplot(my_lm[["residuals"]], geom = "histogram", bins = bins, xlab = 'Residuals Values')
# }

# define function to calculate errors
errors <- function(y, y_hat) 
{
  RSS = sum((y - y_hat)^2)  # residual sum of squares
  TSS = sum((y - mean(y_hat))^2) # total sum of squares
  R2 <- 1 - RSS/TSS  # R squared
  RMSE <- sqrt(mean((y_hat - y)^2)) # root mean square error
  return(list(R2 = R2, RMSE = RMSE))
}


# test the codes ----
# Read data from file into R variable
data1 <- read.csv("TrainData_Group1.csv", header = TRUE)
# create indexes for 75% random sample of the data 
set.seed(101)
indexes = sample(1:nrow(data1), size = 0.75 * nrow(data1)) 
# partition data into traning and test data
train = data1[indexes,] # training data
test = data1[-indexes,] # test data
rm(indexes) # remove the indexes varible

# fit the models ----
# fit the regression model using my_fit_lm() with train data
my_lm_model = my_lm(train[,1:5], train[,6])
# fit the regression model using lm() from R with train data
# where the formula, Y ~ X1 + X2 + X3 + X4 + X5, has an 
# implied intercept term in lm().  To remove the implied intercept,
# use either y ~ x - 1 pr y ~ 0 + x
vanilla_lm_model = lm(Y ~ X1 + X2 + X3 + X4 + X5, train)

# models comparison ----
# components of my_lm_model fitted by my_lm() compare to vanilla_lm_model fitted by lm() 
# coefficients 
print("coefficients from my_lm_model:")
print(my_lm_model[['coeffs']])
print("coefficients from vanillar_lm_model from R:")
print(vanilla_lm_model[['coefficients']])
# accuracy metrics 
print(paste('my_lm_model mse:', my_lm_model$mse)) # mean square error
vanilla_lm_model_mse = mean(vanilla_lm_model[['residuals']]^2)
print(paste("vanilla_lm_model mse: ", vanilla_lm_model_mse)) # mean square error
print(paste('my_lm_model rmse:', my_lm_model$rmse)) # root mean square error
print(paste("vanilla_lm_model rmse: ", sqrt(vanilla_lm_model_mse))) # root mean square error
print(paste('my_lm_model R^2:', my_lm_model$R2)) # R Square
my_vanilla_train_errors = errors(train$Y, predict.my_lm(my_lm_model, train[ , 1:5]))
print(paste("vanilla_lm_model R^2: ", my_vanilla_train_errors$R2)) # R Square

# use models to predict ----
# compare R^2 and RMSE of the two models ----
print('accuracy metrics with test data:')
my_lm_model_errors = errors(test$Y, predict.my_lm(my_lm_model, test[, 1:5]))
vanilla_lm_model_errors = errors(test$Y, predict.lm(vanilla_lm_model, test[ , 1:5]))
print(paste('my_lm_model R^2:', my_lm_model_errors$R2))
print(paste('vanilla_lm_model R^2:', vanilla_lm_model_errors$R2))
print(paste('my_lm_modle RMSE:', my_lm_model_errors$RMSE))
print(paste('vanilla_lm_model RMSE:', vanilla_lm_model_errors$RMSE))

# plots ----
# qplot(my_lm_model[["residuals"]], geom = "histogram", bins = 30, xlab = 'my_lm_model Residuals Values')
# plot_residuals.my_lm(my_lm_model)
# use the fitted model to estimate y_hat of test data
y_hat = predict.my_lm(my_lm_model, test[ ,1:5])
plot(x = test$Y, y = y_hat, pch = "+", col = 'red', main = "My Linear Regression Algorithm: Actual Vs. Predicted",
     xlab = "Actual y", ylab = "Predicted y")

