# Assignment_1 ---- 
# DSSA_5201_Machine_Learning_Fundamentals, Spring_2020
# Write R code from scratch for a K-Means Clustering algorithm using Euclidean distances, which prints 
# the total within-cluster sum of squares and displays a graph using the Elbow Method to determine an 
# appropriate K. Document each step of the code to demonstrate you understand what each line of code 
# does. The code has to describe the steps in creating the model and steps in computing the sum of 
# square errors. For your algorithm, you may use basic statistical functions and graphing functions 
# but NOT machine learning functions such as kmeans(). You can use the attached datasets to train and 
# test your model. Compare your algorithms output of clusters to the algorithms in R using kmeans(). 
# Feel free to experiment with alternative measurements but at least one method must employ Euclidean 
# distances.

# pseudocodes ---- 
# 1. Accept inputs:
#    a) data, numeric matrix of data
#    b) k, the mnuber of centers/clusters to be created
#    c) iter_max, default = 10, the max number of iterations to find new centroids and re-cluster
#    d) dist_measure, default = Euclidean distance, the choice of distance measures
#    e) nstart, default = 25, # of random set of centers to fresh start and get the least tot.withiness
# 2. Select randomly centroids, which are k obsevations from the data set 
# 3. Assign observations to closest centroid, forming clusters
#    Run inner iteration through all observations and assign to the closest centroid, based on the 
#    Euclidean distance between the object and the centroid.
# 4. Update centroids: fine the means of all columns of each cluster
# 5. Change converges flag by evaluate distance bewteen old and new centroids or iterations max out
#     whichever is sooner
# 6. Calculate and save converged results
# 7. If nstart > 1, repeat steps 2 through 6 for nstart times, save new results if better
# 8. Calculate additional outputs
# 9. Return the following:
#    cluster: A vector of integers (from 1:k) indicating the cluster to which each point is allocated.
#    centers: A matrix of cluster centres.
#    totss: The total sum of squared distances.
#    withinss: Vector of within-cluster sum of squares, one component per cluster.
#    tot.withinss: Total within-cluster sum of squares, i.e. sum(withinss).
#    betweenss: The between-cluster sum of squares, i.e. totss-tot.withinss. (ONLY IF I HAVE TIME)
#    size: The number of points in each cluster.



# define my_kmeans() ---- 
my_kmeans = function(dat, k, max_iter = 15, dist_measure = "euclidean", nstart = 30) 
{ # my_kmeans inputs for function
  # 1. Accept my_kmeans inputs ----
  # data = data1 # 
  # k = the mumber of centers/clusters to be created
  # max_iter = the max number of iterations to find new centroids and re-cluster
  # dist_measure = the choice of distance measures
  # nstart = number of times a random set of centers is selected to re-start and get the least 
  #          total within-cluster sum of squared distance of all runs
  X <- as.matrix(dat, nrow(dat), ncol(dat))  # coerce data to a matrix
  k <- as.integer(k)
  nstart <- as.integer(nstart)
  max_iter <- as.integer(max_iter)
  
  # define sum of squared distance function
  ss <- function(Y) {sum(scale(Y, scale = FALSE)^2)}
  
  # initialize variables 
  outer_iter <- 0 # outer_iter counter
  final_run_iter <- 1 # save run_iter when total_withinss first reaches the min
  cluster <- rep(0, nrow(X)) # save assigned cluster in each inner-iteration till converged
  new_cluster <- rep(0, nrow(X)) # save new cluster 
  final_cluster <- rep(0, nrow(X)) # save final cluster after nstart iteration
  centroids <- matrix(0, k, ncol(X)) # centroids matrix
  new_centroids <- matrix(0, k, ncol(X)) # new centroids matrix
  final_centroids <- matrix(0, k, ncol(X)) # final centroids matrix
  size <- rep(0, k) # vector to save the size for each cluster
  withinss <- rep(0, k) # vector to save withincluster ss for all clusters
  new_withinss <- rep(0, k) # vector to save withinss for all new clusters 
  betweenss <- 0 # initialize output for between-cluster sum of squares
  betweenss_over_totss <- "" # initialize betweenss / totss in percent

  # initialize variables for stop criteria for terations
  centroids_dist <- 10e10 # a huge number to start for inner-loop (not used)
  total_withinss <- 10e10 # a huge number to start for inner-loop
  min_total_withinss <- 10e10 # a huge number to start for nstart loop

  # loop for nstart times ----
  for (outer_iter  in 1:nstart)    
  {
    print(paste("outer_iter", outer_iter)) # for debug
    # initialize converged flag for and re-set to FALSE inner-iterations
    converged <- FALSE  # Centroids stop changing for inner_iterations?
    # initialize variables for stop criteria for terations

    # 2. Create initial random centroids ---- 
    # Randomly select k objects (k individual rows) from data as the cluster centers
    centroids <- X[sample(nrow(X), size = k, replace = FALSE), ] # sample k rows from X matrix
    
    # 3. Assign observations to closet centroid ----
    # Run inner iteration through all observations and assign to the closest centroid, based on the 
    # Euclidean distance between the object and the centroid

    # inner iter
    # inner loop till max_iter is reached or total_withinss reached min (converged)
    run_iter <- 0 # initialize run_iter counter
    while ((run_iter < max_iter) & converged == FALSE) 
    { 
      run_iter <- run_iter + 1 # increment run_iter counter
      print(paste("run_iter", run_iter)) # for debug

      # assign data to centroids ----
      for (i in 1:nrow(X)) { # iterate over observations
        min_dist = 10e10 # initialized, a huge distance
        for (c in 1:nrow(centroids)) # iterate over centroids
        {  
          # calculate sum of squared Euclidean distance between observation and centroid
          distance_to_centroid <- sum((centroids[c, ] - X[i, ])^2)
          if (distance_to_centroid < min_dist) # assign to this centroid if it is closer
          { 
            new_cluster[i] = c # save cluster number to new_cluster vector
            min_dist = distance_to_centroid # save the distance as the min_dist
            # print(paste("row :", i, "min_dist:", min_dist)) # for debug
          } 
        } # complete checking all centroids
      } # end iterations over observation     

      # break the loop if there is any loner cluster, which will cause error 
   
      # 4. update centroids ----
      for (i in 1:nrow(centroids))  # for each row of the centroids matrix
      { 
        # apply the mean function to all the columns of X that is filtered by cluster (i) 
        # print(paste("nrow: of cluster", i, nrow(X[new_cluster==i, ]))) # for debug
        try(new_centroids[i, ] <- apply(X[new_cluster == i, ], 2, mean)) # in a try() block
        # the above line is an error-causing line when a cluster has < 2 rows 
        # Putting it in a try() block skips over it with an error message (you can 
        # suppress the error message with the silent=T argument to try), and continues on 
        # with the rest of the scripts. This would likely end the current run_iter.  
      } # update centroids   
      
      # print(new_centroids) # for debug
      
      # 5. change converged flag or let inner-iteration reach max_iter ----
      # calculate new_withinss for each cluster
      for (i in 1:nrow(centroids)) new_withinss[i] <- ss(X[new_cluster == i, ])
      # calculate new_total_withinss
      new_total_withinss = sum(new_withinss) 
      # evaluate condition 
      if (new_total_withinss < total_withinss) # save the results if new_total_withinss is smaller
      { 
        cluster <- new_cluster
        centroids <- new_centroids
        withinss <- new_withinss
        total_withinss <- new_total_withinss
      } else # new_withinss stops getting smaller
      {
        converged <- TRUE 
      } # change converged flag to TRUE to end inner_iteration
      
    } # end loop when max_iter is reached or converged == TRUE
    
    # 6. calculate and save converged results ----
    # save if better than prior 
    # calculate within-cluster sum of squares, withinss, for each cluster
    for (i in 1:nrow(centroids)) withinss[i] <- ss(X[cluster == i, ])
    # calculate total within-cluster sum of squares, total_withinss 
    total_withinss <- sum(withinss)
    print(paste("total_withinss =", total_withinss)) # for debug
    # compare converged results 
    if (total_withinss < min_total_withinss) # if total_withinss is smaller
    {
      min_total_withinss <- total_withinss
      final_cluster <- cluster
      final_withinss <- withinss      
      final_centroids <- centroids 
      final_iter <- run_iter
    } # and save in final result set
    
  } # 7. loop back when nstart > 1 ----

  # 8. calculate additional outputs to return ----
  totss = ss(X) # total sum of squares for all data
  # determine size of each final_cluster
  for (i in 1:k) size[i] <- length(which(final_cluster == i))
  # calculate between-cluster sum of squares
  betweenss	<- totss - min_total_withinss
  # calculate betweenss / totss (round to 1 digit and present in percent)
  betweenss_over_totss = paste(format(round(betweenss/totss*100, 1), nsmall = 1), '%')
                
  # 9. return function outputs ----
  return(list(cluster = final_cluster, centers = final_centroids, totss = totss,
              withinss = final_withinss, tot.withinss = min_total_withinss, 
              betweenss = betweenss, size = size, iter = final_run_iter, 
              betweenss_over_totss = betweenss_over_totss)) 
} # my_kmeans ends

# Run my_kmeans() ---- #################################################################
# import library ---- 
library(readr) 

# read data from file ---- 
dat <- read_csv("KMeansData_Group1.csv", col_names = FALSE)   # from csv file
# X <- as.matrix(dat)

# evaluate for optimal for k ----
# run kmeans() from R in each loop from K = 1 through 8, saving the tot.withinss for each loop
Nstart = 35
Max_iter = 10
K <- c() # initialize vector for Elbow Graph
TOTAL_WITHINSS <- c() # initialize vector for Elbow Graph, cl$tot.withinss
for (i in 2:8)
{
  mycl <- my_kmeans(dat, k = i, max_iter = Max_iter, nstart = Nstart)
  title = paste("my_kmeans Clustering:  k = ", as.character(i),
                ",  nstart = ", as.character(Nstart))
  plot(dat, col = mycl$cluster, main = title) # all data points colored by cluster
  points(mycl$centers, col = 80, pch = 13, cex = 7, lwd = 3)  # the center of each cluster
  K <- c(K, i)
  TOTAL_WITHINSS <- c(TOTAL_WITHINSS, mycl$tot.withinss)
}

# create elbow graph ----
title = paste("my_kmeans, Elbow Graph, nstart = ", as.character(Nstart))
plot(K, TOTAL_WITHINSS, main = title)
lines(K, TOTAL_WITHINSS)

# run my_kmeans() with optimal values for k and nstart ----
k = 5
Nstart = 25
Max_iter = 10
mycl <- my_kmeans(dat, k, max_iter = Max_iter, nstart = Nstart)
# plot data graph colored by clustter, showing centers
title = paste("my_kmeans, k = ", as.character(k), ",  nstart = ", as.character(Nstart))
plot(dat, col = mycl$cluster, main = title)  # all data colored by cluster
points(mycl$centers, col = 80, pch = 13, cex = 7, lwd = 3)  # the center of each cluster
# print all outputs 
print(mycl)
print(str(mycl))

# compare to results from R kmeans
cl <- kmeans(dat, centers = k, iter.max = Max_iter, nstart = Nstart)
# plot data graph colored by clustter, showing centers
title = paste("R kmeans, k = ", as.character(k), ",  nstart = ", as.character(Nstart))
plot(dat, col = cl$cluster, main = title) # all data colored by cluster
points(cl$centers, col = 80, pch = 13, cex = 7, lwd = 3)  # the center of each cluster
# print all outputs 
print(cl)
print(str(cl))

