################################################################################
#######.  EXPERIMENTATION FOR SECTION 4.2 of the paper:
#######. 
#######.  A Simple Approach to Multiple Instance Learning:
#######.  Controlling the False Positive Rate
#######.     
#######.  by Rosario Delgado
#######.  September, 2024
################################################################################


# Cleaning environment
rm(list=ls());gc()


library(ggplot2)
library(tidyverse)
library(milr)

source("Algorithms.R")
source("mat_square.R")

################################################################################
####. Performance metrics:
####.   
####. F1-score:

F1.score<-function(A) # square 2 x 2 matrix. Classes 0, 1. Focused on class 0 
{F<-2*A[1,1]/(2*A[1,1]+A[1,2]+A[2,1])
return(F)}

####.  Accuracy:

Accuracy<-function(A) # square 2 x 2 matrix. 
{Acc<-(A[1,1]+A[2,2])/sum(A)
return(Acc)}

####.  Balance Accuracy

BA<-function(A) # square 2 x 2 matrix. 
{Balance.Accuracy<-(A[1,1]/(A[1,1]+A[1,2])+A[2,2]/(A[2,2]+A[2,1]))/2
return(Balance.Accuracy)}

################################################################################

################################################################################
####### We consider one of the two datasets considered by Dietterich et al. 
#######
####### Musk-2: 102 molecules=bags (39 positive and 63 negative), 
#######         6598 instances and 166 attributes. 
#######
#######. download from: https://www.uco.es/kdos/momil
################################################################################

####### input 10 train and 10 test sets 
####### j=1,..., 10 indicates the fold

names <- paste0("V", 1:166)   # Feature names for musk2 

# ignore @ lines by defining @ as the comment character

################################################################################
############## STEP 0 a): LOADING AND PREPROCESSING THE TRAINING DATA SETS
##############.  As an example, we do it for the Procedure 1
################################################################################

data.tra<-list()  # 10 train datasets

### training set j, j=1,..., 10

File <- "nweka-musk2-10-jtra.arff"
table.data<-read.table(File, header = FALSE, sep = ",", comment.char = "@", fill=TRUE)
table.data<-table.data[-1,]

dt1 <- table.data
colnames(dt1)<-c("bag","features","Class")

dt2<-dt1 %>%
  separate_rows(features, sep = ",", convert = F) %>%
  mutate(id = rep(names, times = nrow(.) / 166)) %>% 
  pivot_wider(names_from = id, values_from = features, values_fn = list) %>% 
  unnest(cols = names)  


dt2<-as.data.frame(dt2)

xxx<-separate(dt2, V1, c("Remove", "V1.new"),sep="n")
for (i in 1:dim(xxx)[1])
{if (is.na(xxx$V1.new[i])==TRUE)
{xxx$V1.new[i]<-xxx$Remove[i]}
}

xxx.2<-separate(xxx, V1.new, c("Remove.2", "V1.new.2"),sep="-")
for (i in 1:dim(xxx.2)[1])
{if (is.na(xxx.2$V1.new.2[i])==TRUE)
{xxx.2$V1.new.2[i]<-xxx.2$Remove.2[i]}
}


data.tra[[i]]<-xxx.2[,-c(3,4)]


##########

for (j in 1:10)
{colnames(data.tra[[j]])[3]<-"V1"
 data.tra[[j]][,-c(1,2)]<-apply(data.tra[[j]][,-c(1,2)], 2, as.numeric)
}

################################################################################
## 
## data.tra list with 10 training datasets
## 
################################################################################

N.tra<-vector()     # number of bags for any train datset

for (j in 1:10)
{
  N.tra[j]<-length(unique(data.tra[[j]]$bag))
}


################################################################################
################################################################################
################################################################################

################################################################################
####   STEP 1: LEARNING PREDICTIVE MODELS WITH TRAIN DATASETS, AND 
####           ESTIMATING q = false positive rate (instance level) 
################################################################################
bag.name.tra<-list()  # bags' names in the training datasets

for (j in 1:10)
{bag.name.tra[[j]]<-unique(data.tra[[j]]$bag)}


Y.tra<-list()     # actual class (0/1) for each bag in each training set

for (j in 1:10)
{ Y.tra[[j]]<-vector()
  for (k in 1:N.tra[j])  
   {if (sum(data.tra[[j]]$Class[which(data.tra[[j]]$bag==bag.name.tra[[j]][k])])>0)
   {Y.tra[[j]][k]<-1} else {Y.tra[[j]][k]<-0}
   }
}

##

bags.tra.0<-list()  # bags numbers in training sets with actual class = 0
bags.tra.1<-list()  # bags numbers in training sets with actual class = 1

for (j in 1:10)
{bags.tra.0[[j]]<-which(Y.tra[[j]]==0)
 bags.tra.1[[j]]<-which(Y.tra[[j]]==1)
}

##


size.bag.tra<-list()  # bags' sizes in the training datasets
instances.tra.bag.0<-list()  # instances in training sets with actual class = 0
instances.tra.bag.1<-list()  # instances in training sets with actual class = 0

for (j in 1:10)
{instances.tra.bag.0[[j]]<-which(data.tra[[j]]$bag %in% bag.name.tra[[j]][bags.tra.0[[j]]])
 instances.tra.bag.1[[j]]<-which(data.tra[[j]]$bag %in% bag.name.tra[[j]][bags.tra.1[[j]]])
 size.bag.tra[[j]]<-vector()
 
 for (k in 1:N.tra[[j]])
  {size.bag.tra[[j]][k]<-length(which(data.tra[[j]]$bag==bag.name.tra[[j]][k]))
  }
}



sum.instances.tra.bag.0<-vector()   # sum of number of instances in bags of actual class = 0
sum.instances.tra.bag.1<-vector()   # sum of number of instances in bags of actual class = 1
for (j in 1:10)
{sum.instances.tra.bag.0[j]<-sum(size.bag.tra[[j]][bags.tra.0[[j]]])
 sum.instances.tra.bag.1[j]<-sum(size.bag.tra[[j]][bags.tra.1[[j]]])
}

#####

X.tra<-list() # matrices for each milr model, learned from any training dataset

for (j in 1:10)
{X.tra[[j]]<-as.matrix(data.tra[[j]][,-c(1,2)])
}


#####
##  We use both milr (Chen et al., 2016) and softmax (Xu and Frank, 2004) approaches. 
##  milr::softmax function, with alpha=0 and with alpha=3, is used to find a 
##  preliminar model from which perform feature selection, and then apply 
##  milr::milr function for final model estimation with variable selection.
##  
##  fitted values are obtained with milr::fitted function, both at the bag and at the 
##  instance levels.
##
##  For the milr function, we use a very small lambda = 1e-7 so that it do the estimation 
##   without evaluating the Hessian matrix. 


milr.model.s0<-list()   # the predictive milr models
selected.s0<-list()
milr.model.s3<-list()   # the predictive milr models
selected.s3<-list()

for (j in 1:10)
{fit.s0 <- softmax(data.tra[[j]]$Class, X.tra[[j]], 
                   as.numeric(as.factor(data.tra[[j]]$bag)),
                   alpha = 0, control = list(maxit = 500)) # Xu and Frank, 2004

fit.s3 <- softmax(data.tra[[j]]$Class, X.tra[[j]], 
                  as.numeric(as.factor(data.tra[[j]]$bag)),
                  alpha = 3, control = list(maxit = 500)) # Ray and Craven, 2005

###############
### Select variables with absolute value of coefficient > 0.005 but avoid 
### to have only 1 (milr gives error) or > 10
##############

selected.s0[[j]] <- names(which(abs(coef(fit.s0)[-1L]) > 0.005)) 

step<-0
while(length(selected.s0[[j]])<2)
{selected.s0[[j]] <- names(which(abs(coef(fit.s0)[-1L]) > 0.005-step))
 step<-step+0.001}

step<-0
while (length(selected.s0[[j]])>10)
{selected.s0[[j]] <- names(which(abs(coef(fit.s0)[-1L]) > 0.005+step))
 step<-step+0.001}

##

selected.s3[[j]] <- names(which(abs(coef(fit.s3)[-1L]) > 0.005)) 

step<-0
while(length(selected.s3[[j]])<2)
{selected.s3[[j]] <- names(which(abs(coef(fit.s3)[-1L]) > 0.005-step))
step<-step+0.001}

step<-0
while (length(selected.s3[[j]])>10)
{selected.s3[[j]] <- names(which(abs(coef(fit.s3)[-1L]) > 0.005+step))
step<-step+0.001}


##########################
## The predictive milr model with the selected features
##
#########################
  
milr.model.s0[[j]]<- milr::milr(data.tra[[j]]$Class, X.tra[[j]][,selected.s0[[j]]], 
                                as.numeric(as.factor(data.tra[[j]]$bag)),lambda = 1e-7, 
                                maxit = 500) 

milr.model.s3[[j]]<- milr::milr(data.tra[[j]]$Class, X.tra[[j]][,selected.s3[[j]]], 
                                as.numeric(as.factor(data.tra[[j]]$bag)),lambda = 1e-7, 
                                maxit = 500) 

print(j)  
}


#####

pred.instances.tra.s0<-list()   # the predictions al the instance level with milr.models
pred.instances.tra.s3<-list()

for (j in 1:10)
{pred.instances.tra.s0[[j]]<-fitted(milr.model.s0[[j]], type = "instance")
pred.instances.tra.s3[[j]]<-fitted(milr.model.s3[[j]], type = "instance")
}
  
#####

epsilon=1e-3

q.est.s0<-vector()    # estimations of q = false positive rate
q.est.s3<-vector() 

for (j in 1:10)   # we use epsilon to be sure that q.est<0.5
  {q.est.s0[j]<-
     min(sum(pred.instances.tra.s0[[j]][unlist(instances.tra.bag.0[j])])/sum.instances.tra.bag.0[j],0.5-epsilon)
  q.est.s3[j]<-
    min(sum(pred.instances.tra.s3[[j]][unlist(instances.tra.bag.0[j])])/sum.instances.tra.bag.0[j],0.5-epsilon)
  }


################################################################################
########
########  Prediction for bags in training dataset using the milr.model directly
########
################################################################################
table.fitted.bag.s0<-list()
table.fitted.bag.s3<-list()

for (j in 1:10)
{table.fitted.bag.s0[[j]]<-table(DATA = Y.tra[[j]], FIT_MILR = fitted(milr.model.s0[[j]], type = "bag"))
 table.fitted.bag.s3[[j]]<-table(DATA = Y.tra[[j]], FIT_MILR = fitted(milr.model.s3[[j]], type = "bag"))
}

table.fitted.bag.s0
table.fitted.bag.s3

################################################################################
#########
### Decision about the bags in training dataset using standard procedure (c=1)
### from the predictions obtained for the instances with the milr.model
########
################################################################################
pred.bag.tra.c.1.s0<-list()
pred.bag.tra.c.1.s3<-list()

for (j in 1:10)
{pred.bag.tra.c.1.s0[[j]]<-vector()
 pred.bag.tra.c.1.s3[[j]]<-vector()
 for (k in 1:N.tra[j])
 {pred.bag.tra.c.1.s0[[j]][k]<-if_else(sum(pred.instances.tra.s0[[j]][which(data.tra[[j]]$bag==bag.name.tra[[j]][k])])>0,1,0)
  pred.bag.tra.c.1.s3[[j]][k]<-if_else(sum(pred.instances.tra.s3[[j]][which(data.tra[[j]]$bag==bag.name.tra[[j]][k])])>0,1,0)
 }
}


table.fitted.bag.c.1.s0<-list()
table.fitted.bag.c.1.s3<-list()
for (j in 1:10)
{table.fitted.bag.c.1.s0[[j]]<-table(DATA = Y.tra[[j]], pred.bag.tra.c.1.s0[[j]])
 table.fitted.bag.c.1.s3[[j]]<-table(DATA = Y.tra[[j]], pred.bag.tra.c.1.s3[[j]])
}
table.fitted.bag.c.1.s0
table.fitted.bag.c.1.s3


################################################################################
###### Metrics for fitted bag labels, and predicted bag labels with c=1
###### from the predicted instance labels, both using the milr.model
################################################################################

F1.score.fitted.bag.s0<-vector()  # F1 score for bag directly fitted by milr with s(0)
F1.score.fitted.bag.c.1.s0<-vector() # F1 score for the bag using c=1 with s(0) 
F1.score.fitted.bag.s3<-vector() # similar with s(3)
F1.score.fitted.bag.c.1.s3<-vector()  # similar with s(3=)

for (j in 1:10)
{F1.score.fitted.bag.s0[j]<-F1.score(mat.square(table.fitted.bag.s0[[j]],c("0","1")))
 F1.score.fitted.bag.c.1.s0[j]<-F1.score(mat.square(table.fitted.bag.c.1.s0[[j]],c("0","1")))
 F1.score.fitted.bag.s3[j]<-F1.score(mat.square(table.fitted.bag.s3[[j]],c("0","1")))
 F1.score.fitted.bag.c.1.s3[j]<-F1.score(mat.square(table.fitted.bag.c.1.s3[[j]],c("0","1")))
 }

F1.score.fitted.bag.s0
F1.score.fitted.bag.c.1.s0
F1.score.fitted.bag.s3
F1.score.fitted.bag.c.1.s3

##

Accuracy.fitted.bag.s0<-vector()   # The same with Accuracy
Accuracy.fitted.bag.c.1.s0<-vector()
Accuracy.fitted.bag.s3<-vector()
Accuracy.fitted.bag.c.1.s3<-vector()

for (j in 1:10)
{Accuracy.fitted.bag.s0[j]<-Accuracy(mat.square(table.fitted.bag.s0[[j]],c("0","1")))
Accuracy.fitted.bag.c.1.s0[j]<-Accuracy(mat.square(table.fitted.bag.c.1.s0[[j]],c("0","1")))
Accuracy.fitted.bag.s3[j]<-Accuracy(mat.square(table.fitted.bag.s3[[j]],c("0","1")))
Accuracy.fitted.bag.c.1.s3[j]<-Accuracy(mat.square(table.fitted.bag.c.1.s3[[j]],c("0","1")))
}

Accuracy.fitted.bag.s0
Accuracy.fitted.bag.c.1.s0
Accuracy.fitted.bag.s3
Accuracy.fitted.bag.c.1.s3

##

BA.fitted.bag.s0<-vector()  # The same with Balance Accuracy (BA)
BA.fitted.bag.c.1.s0<-vector()
BA.fitted.bag.s3<-vector()
BA.fitted.bag.c.1.s3<-vector()

for (j in 1:10)
{BA.fitted.bag.s0[j]<-BA(mat.square(table.fitted.bag.s0[[j]],c("0","1")))
BA.fitted.bag.c.1.s0[j]<-BA(mat.square(table.fitted.bag.c.1.s0[[j]],c("0","1")))
BA.fitted.bag.s3[j]<-BA(mat.square(table.fitted.bag.s3[[j]],c("0","1")))
BA.fitted.bag.c.1.s3[j]<-BA(mat.square(table.fitted.bag.c.1.s3[[j]],c("0","1")))
}

BA.fitted.bag.s0
BA.fitted.bag.c.1.s0
BA.fitted.bag.s3
BA.fitted.bag.c.1.s3

################################################################################
#########
### Decision about the bags in training dataset using new procedure (c optimal, c*)
### from the predictions obtained for the instances with the milr.model
########
################################################################################

pi.estimated.s0<-list()  # estimated pi = proportion of positively classified instances 
                         # in each bag of each fold with milr model using s(0) for 
                         # feature selection
pi.estimated.s3<-list()  # the same with s(3)

for (j in 1:10)
{pi.estimated.s0[[j]]<-vector()
 pi.estimated.s3[[j]]<-vector()
 for (k in 1:N.tra[j])    # k es el bag, de 1 a 91
 { pi.estimated.s0[[j]][k]=sum(pred.instances.tra.s0[[j]][which(data.tra[[j]]$bag==bag.name.tra[[j]][k])])/
   length(pred.instances.tra.s0[[j]][which(data.tra[[j]]$bag==bag.name.tra[[j]][k])])
 pi.estimated.s3[[j]][k]=sum(pred.instances.tra.s3[[j]][which(data.tra[[j]]$bag==bag.name.tra[[j]][k])])/
   length(pred.instances.tra.s3[[j]][which(data.tra[[j]]$bag==bag.name.tra[[j]][k])])
 }
}


########################
#### Fix parameteres

alpha<-0.10
beta<-0.10

#########################
##### Introduce the values of p and q. The values for q are obtained from q.estimated

v.p<-seq(from = 0.001, to = 0.2, by=0.01)  # 20 values for p = false negative rate

v.q.s0<-list() # some values for q = false positive rate, around the estimated value
v.q.s3<-list()
for (j in 1:10)
{ if(q.est.s0[j]==0)
 {v.q.s0[[j]]<-seq(from = epsilon, to = min(50*epsilon,0.5-epsilon), by=0.01)} else { 
  v.q.s0[[j]]<-seq(from = q.est.s0[j]-q.est.s0[j]/3, 
                   to = min(q.est.s0[j]+q.est.s0[j]/3,0.5-epsilon), by=0.01)
  } 
  if(q.est.s3[j]==0)
  {v.q.s3[[j]]<-seq(from = epsilon, to = min(50*epsilon,0.5-epsilon), by=0.01)} else { 
    v.q.s3[[j]]<-seq(from = q.est.s3[j]-q.est.s3[j]/3, 
                     to = min(q.est.s3[j]+q.est.s3[j]/3,0.5-epsilon), by=0.01)
  }  
} 


vector.p.q.s0<-list() # combined values for p and q
vector.p.q.s3<-list()
for (j in 1:10)
  {vector.p.q.s0[[j]]<-data.frame(p = v.p, q = rep(v.q.s0[[j]],each=length(v.p))) 
  vector.p.q.s3[[j]]<-data.frame(p = v.p, q = rep(v.q.s3[[j]],each=length(v.p))) 
  } 

################### 
###### Estimation of m+ and m- using Algorithm 2
###### implemented with the function "Algorithms::m.plus.minus"


m.plus.tra.s0<-list()
m.plus.tra.s3<-list()
m.minus.tra.s0<-list()
m.minus.tra.s3<-list()

for (j in 1:10)
{ m.plus.tra.s0[[j]]<-list()
  m.minus.tra.s0[[j]]<-list()  
  m.plus.tra.s3[[j]]<-list()
  m.minus.tra.s3[[j]]<-list() 

  for (i in 1:dim(vector.p.q.s0[[j]])[1])
  { m.plus.tra.s0[[j]][[i]]<-vector()
    m.minus.tra.s0[[j]][[i]]<-vector()
    for (k in 1:N.tra[j])
    {m.plus.tra.s0[[j]][[i]][k]<-as.numeric(m.plus.minus(size.bag.tra[[j]][k],
                                  vector.p.q.s0[[j]][i,1],vector.p.q.s0[[j]][i,2],
                                  pi.estimated.s0[[j]][k])[[2]][1])
     m.minus.tra.s0[[j]][[i]][k]<-as.numeric(m.plus.minus(size.bag.tra[[j]][k],
                                  vector.p.q.s0[[j]][i,1],vector.p.q.s0[[j]][i,2],
                                  pi.estimated.s0[[j]][k])[[2]][2])
    }
  }

  for (i in 1:dim(vector.p.q.s3[[j]])[1])
  { m.plus.tra.s3[[j]][[i]]<-vector()
  m.minus.tra.s3[[j]][[i]]<-vector()
  for (k in 1:N.tra[j])
  {m.plus.tra.s3[[j]][[i]][k]<-as.numeric(m.plus.minus(size.bag.tra[[j]][k],
                                                       vector.p.q.s3[[j]][i,1],vector.p.q.s3[[j]][i,2],
                                                       pi.estimated.s3[[j]][k])[[2]][1])
  m.minus.tra.s3[[j]][[i]][k]<-as.numeric(m.plus.minus(size.bag.tra[[j]][k],
                                                       vector.p.q.s3[[j]][i,1],vector.p.q.s3[[j]][i,2],
                                                       pi.estimated.s3[[j]][k])[[2]][2])
  }
  }
  
}


#########################
#### Find the optimal value of c, c* with Algorithm 1, 
#### implemented with the funcion "Algorithms::optimal.c"

result.opt.c.s0<-list()
result.opt.c.s3<-list()

for (j in 1:10)
{result.opt.c.s0[[j]]<-list()
 result.opt.c.s3[[j]]<-list()
 
 for (i in 1:dim(vector.p.q.s0[[j]])[1])   
 {result.opt.c.s0[[j]][[i]]<-list()
  for (k in 1:N.tra[j])
  {result.opt.c.s0[[j]][[i]][[k]]<-optimal.c(size.bag.tra[[j]][k],
                                vector.p.q.s0[[j]][i,1],vector.p.q.s0[[j]][i,2],
                                alpha,beta,
                                m.plus.tra.s0[[j]][[i]][k],m.minus.tra.s0[[j]][[i]][k])
  }
 }

 for (i in 1:dim(vector.p.q.s3[[j]])[1])   
 {result.opt.c.s3[[j]][[i]]<-list()
 for (k in 1:N.tra[j])
 {result.opt.c.s3[[j]][[i]][[k]]<-optimal.c(size.bag.tra[[j]][k],
                                            vector.p.q.s3[[j]][i,1],vector.p.q.s3[[j]][i,2],
                                            alpha,beta,
                                            m.plus.tra.s3[[j]][[i]][k],m.minus.tra.s3[[j]][[i]][k])
 }
 }
 
 }


##################
### Decision about the bags with c-rule, c given by result.opt.c


pred.bag.tra.c.opt.s0<-list()    # 1 = +, 0 = -
pred.bag.tra.c.opt.s3<-list()

for (j in 1:10)
 {pred.bag.tra.c.opt.s0[[j]]<-list() 
  pred.bag.tra.c.opt.s3[[j]]<-list() 

  for (i in 1:dim(vector.p.q.s0[[j]])[1])  
   {pred.bag.tra.c.opt.s0[[j]][[i]]<-vector()
    for (k in 1:N.tra[j])
     {if (is.null(result.opt.c.s0[[j]][[i]][[k]][5,2])==TRUE)
       {pred.bag.tra.c.opt.s0[[j]][[i]][k]<-NA} else {
       if(is.na(as.numeric(result.opt.c.s0[[j]][[i]][[k]][5,2]))==FALSE)
         {pred.bag.tra.c.opt.s0[[j]][[i]][k]<-if_else(
              sum(pred.instances.tra.s0[[j]][which(data.tra[[j]]$bag==bag.name.tra[[j]][k])]) >
                as.numeric(result.opt.c.s0[[j]][[i]][[k]][5,2])-1,1,0)
         } else {pred.bag.tra.c.opt.s0[[j]][[i]][k]<-NA}
       }
     }
  }
  
  
  for (i in 1:dim(vector.p.q.s3[[j]])[1])  
  {pred.bag.tra.c.opt.s3[[j]][[i]]<-vector()
  for (k in 1:N.tra[j])
  {if (is.null(result.opt.c.s3[[j]][[i]][[k]][5,2])==TRUE)
  {pred.bag.tra.c.opt.s3[[j]][[i]][k]<-NA} else {
    if(is.na(as.numeric(result.opt.c.s3[[j]][[i]][[k]][5,2]))==FALSE)
    {pred.bag.tra.c.opt.s3[[j]][[i]][k]<-if_else(
      sum(pred.instances.tra.s3[[j]][which(data.tra[[j]]$bag==bag.name.tra[[j]][k])]) >
        as.numeric(result.opt.c.s3[[j]][[i]][[k]][5,2])-1,1,0)
    } else {pred.bag.tra.c.opt.s3[[j]][[i]][k]<-NA}
  }
  }
  }
  
}

####################################
##### The confusion matrices


table.bag.tra.c.opt.s0<-list()
table.bag.tra.c.opt.s3<-list()

for (j in 1:10)
 {table.bag.tra.c.opt.s0[[j]]<-list()
  table.bag.tra.c.opt.s3[[j]]<-list()
  
  for (i in 1:dim(vector.p.q.s0[[j]])[1])  
   {table.bag.tra.c.opt.s0[[j]][[i]]<-table(DATA = Y.tra[[j]], pred.bag.tra.c.opt.s0[[j]][[i]])
  }

  for (i in 1:dim(vector.p.q.s3[[j]])[1])  
  {table.bag.tra.c.opt.s3[[j]][[i]]<-table(DATA = Y.tra[[j]], pred.bag.tra.c.opt.s3[[j]][[i]])
  }
  
}

table.bag.tra.c.opt.s0
table.bag.tra.c.opt.s3


################################################################################
###### Metrics computation for the predicted bag labels with new procedure, c optimal (c*)
###### from the predicted instance labels using the milr.model
################################################################################
## Apply "Algorithms::mat.square" to complement the confusion matrices if needed

F1.score.bag.tra.c.opt.s0<-list()
F1.score.bag.tra.c.opt.s3<-list()

for (j in 1:10)
 {F1.score.bag.tra.c.opt.s0[[j]]<-vector()
 F1.score.bag.tra.c.opt.s3[[j]]<-vector()
  
 for (i in 1:dim(vector.p.q.s0[[j]])[1])
   {F1.score.bag.tra.c.opt.s0[[j]][i]<-F1.score(mat.square(table.bag.tra.c.opt.s0[[j]][[i]],c("0","1")))
   }
  
 for (i in 1:dim(vector.p.q.s3[[j]])[1])
 {F1.score.bag.tra.c.opt.s3[[j]][i]<-F1.score(mat.square(table.bag.tra.c.opt.s3[[j]][[i]],c("0","1")))
 }
 
 }

F1.score.bag.tra.c.opt.s0
F1.score.bag.tra.c.opt.s3
mean.F1.score.bag.c.opt.s0<-unlist(lapply(F1.score.bag.tra.c.opt.s0,mean))
mean.F1.score.bag.c.opt.s3<-unlist(lapply(F1.score.bag.tra.c.opt.s3,mean))

F1.score.fitted.bag.s0
F1.score.fitted.bag.c.1.s0
F1.score.fitted.bag.s3
F1.score.fitted.bag.c.1.s3

##

Accuracy.bag.tra.c.opt.s0<-list()
Accuracy.bag.tra.c.opt.s3<-list()

for (j in 1:10)
{Accuracy.bag.tra.c.opt.s0[[j]]<-vector()
 Accuracy.bag.tra.c.opt.s3[[j]]<-vector()
 
for (i in 1:dim(vector.p.q.s0[[j]])[1])
{Accuracy.bag.tra.c.opt.s0[[j]][i]<-Accuracy(mat.square(table.bag.tra.c.opt.s0[[j]][[i]],c("0","1")))
}

for (i in 1:dim(vector.p.q.s3[[j]])[1])
 {Accuracy.bag.tra.c.opt.s3[[j]][i]<-Accuracy(mat.square(table.bag.tra.c.opt.s3[[j]][[i]],c("0","1")))
 }
 
}

Accuracy.bag.tra.c.opt.s0
Accuracy.bag.tra.c.opt.s3
mean.Accuracy.bag.c.opt.s0<-unlist(lapply(Accuracy.bag.tra.c.opt.s0,mean))
mean.Accuracy.bag.c.opt.s3<-unlist(lapply(Accuracy.bag.tra.c.opt.s3,mean))

Accuracy.fitted.bag.s0
Accuracy.fitted.bag.c.1.s0
Accuracy.fitted.bag.s3
Accuracy.fitted.bag.c.1.s3

##

BA.bag.tra.c.opt.s0<-list()
BA.bag.tra.c.opt.s3<-list()

for (j in 1:10)
{BA.bag.tra.c.opt.s0[[j]]<-vector()
 BA.bag.tra.c.opt.s3[[j]]<-vector()

for (i in 1:dim(vector.p.q.s0[[j]])[1])
{BA.bag.tra.c.opt.s0[[j]][i]<-BA(mat.square(table.bag.tra.c.opt.s0[[j]][[i]],c("0","1")))
}
 
for (i in 1:dim(vector.p.q.s3[[j]])[1])
 {BA.bag.tra.c.opt.s3[[j]][i]<-BA(mat.square(table.bag.tra.c.opt.s3[[j]][[i]],c("0","1")))
 }
 
}

BA.bag.tra.c.opt.s0
BA.bag.tra.c.opt.s3
mean.BA.bag.c.opt.s0<-unlist(lapply(BA.bag.tra.c.opt.s0,mean))
mean.BA.bag.c.opt.s3<-unlist(lapply(BA.bag.tra.c.opt.s3,mean))

BA.fitted.bag.s0
BA.fitted.bag.c.1.s0
BA.fitted.bag.s3
BA.fitted.bag.c.1.s3

################################################################################
###### Statistical tests for metrics comparison for the predicted bag labels 
###### with new procedure c optimal (c*),
###### fitted bag labels and predicted bag labels with c=1, from the predicted instance labels 
###### obtained using the milr.model
################################################################################


shapiro.test(mean.F1.score.bag.c.opt.s0-F1.score.fitted.bag.c.1.s0)
t.test(mean.F1.score.bag.c.opt.s0,F1.score.fitted.bag.c.1.s0,paired=TRUE,alternative="greater")
wilcox.test(mean.F1.score.bag.c.opt.s0,F1.score.fitted.bag.c.1.s0,paired=TRUE,alternative="greater")

shapiro.test(mean.F1.score.bag.c.opt.s3-F1.score.fitted.bag.c.1.s3)
t.test(mean.F1.score.bag.c.opt.s3,F1.score.fitted.bag.c.1.s3,paired=TRUE,alternative="greater")
wilcox.test(mean.F1.score.bag.c.opt.s3,F1.score.fitted.bag.c.1.s3,paired=TRUE,alternative="greater")


shapiro.test(mean.F1.score.bag.c.opt.s0-F1.score.fitted.bag.s0)
t.test(mean.F1.score.bag.c.opt.s0,F1.score.fitted.bag.s0,paired=TRUE,alternative="greater")
wilcox.test(mean.F1.score.bag.c.opt.s0,F1.score.fitted.bag.s0,paired=TRUE,alternative="greater")

shapiro.test(mean.F1.score.bag.c.opt.s3-F1.score.fitted.bag.s3)
t.test(mean.F1.score.bag.c.opt.s3,F1.score.fitted.bag.s3,paired=TRUE,alternative="greater")
wilcox.test(mean.F1.score.bag.c.opt.s3,F1.score.fitted.bag.s3,paired=TRUE,alternative="greater")


### 
### 

shapiro.test(mean.Accuracy.bag.c.opt.s0-Accuracy.fitted.bag.c.1.s0)
t.test(mean.Accuracy.bag.c.opt.s0,Accuracy.fitted.bag.c.1.s0,paired=TRUE,alternative="greater")
wilcox.test(mean.Accuracy.bag.c.opt.s0,Accuracy.fitted.bag.c.1.s0,paired=TRUE,alternative="greater")

shapiro.test(mean.Accuracy.bag.c.opt.s3-Accuracy.fitted.bag.c.1.s3)
t.test(mean.Accuracy.bag.c.opt.s3,Accuracy.fitted.bag.c.1.s3,paired=TRUE,alternative="greater")
wilcox.test(mean.Accuracy.bag.c.opt.s3,Accuracy.fitted.bag.c.1.s3,paired=TRUE,alternative="greater")

shapiro.test(mean.Accuracy.bag.c.opt.s0-Accuracy.fitted.bag.s0)
t.test(mean.Accuracy.bag.c.opt.s0,Accuracy.fitted.bag.s0,paired=TRUE,alternative="greater")
wilcox.test(mean.Accuracy.bag.c.opt.s0,Accuracy.fitted.bag.s0,paired=TRUE,alternative="greater")

shapiro.test(mean.Accuracy.bag.c.opt.s3-Accuracy.fitted.bag.s3)
t.test(mean.Accuracy.bag.c.opt.s3,Accuracy.fitted.bag.s3,paired=TRUE,alternative="greater")
wilcox.test(mean.Accuracy.bag.c.opt.s3,Accuracy.fitted.bag.s3,paired=TRUE,alternative="greater")


### 
### 

shapiro.test(mean.BA.bag.c.opt.s0-BA.fitted.bag.c.1.s0)
t.test(mean.BA.bag.c.opt.s0,BA.fitted.bag.c.1.s0,paired=TRUE,alternative="greater")
wilcox.test(mean.BA.bag.c.opt.s0,BA.fitted.bag.c.1.s0,paired=TRUE,alternative="greater")

shapiro.test(mean.BA.bag.c.opt.s3-BA.fitted.bag.c.1.s3)
t.test(mean.BA.bag.c.opt.s3,BA.fitted.bag.c.1.s3,paired=TRUE,alternative="greater")
wilcox.test(mean.BA.bag.c.opt.s3,BA.fitted.bag.c.1.s3,paired=TRUE,alternative="greater")

shapiro.test(mean.BA.bag.c.opt.s0-BA.fitted.bag.s0)
t.test(mean.BA.bag.c.opt.s0,BA.fitted.bag.s0,paired=TRUE,alternative="greater")
wilcox.test(mean.BA.bag.c.opt.s0,BA.fitted.bag.s0,paired=TRUE,alternative="greater")

shapiro.test(mean.BA.bag.c.opt.s3-BA.fitted.bag.s3)
t.test(mean.BA.bag.c.opt.s3,BA.fitted.bag.s3,paired=TRUE,alternative="greater")
wilcox.test(mean.BA.bag.c.opt.s3,BA.fitted.bag.s3,paired=TRUE,alternative="greater")


##########################################################
##########################################################
### 
### Step 2) Now we make some boxplots for the the training sets (fitted values), 
### with the new procedure
### c.opt (c*), since up to now we deal with the mean values, 
### for the results obtained for any combination of values of p (the same for all 
### the folds) and q (different values for any fold, estimated). 
### dim(vector.p.q[[j]])[1] is the number of combinations for fold j
###
###########################################################
###########################################################

## F1.score
Diff.F1.score.bag.tra.s0<-list()
Diff.F1.score.bag.tra.s3<-list()
for (j in 1:10)
{Diff.F1.score.bag.tra.s0[[j]]<-F1.score.bag.tra.c.opt.s0[[j]]-F1.score.fitted.bag.c.1.s0[j]
 Diff.F1.score.bag.tra.s3[[j]]<-F1.score.bag.tra.c.opt.s3[[j]]-F1.score.fitted.bag.c.1.s3[j]}

boxplot.Diff.F1.score.bag.tra <- data.frame(Diff.F1.score.proc1 = c(unlist(Diff.F1.score.bag.tra.s0),
                                                                        unlist(Diff.F1.score.bag.tra.s3)), 
                training.set = c(as.factor(rep(1:10,times = sapply(Diff.F1.score.bag.tra.s0,length))), 
                                 as.factor(rep(1:10,times = sapply(Diff.F1.score.bag.tra.s3,length)))), 
                var.selec = c(as.factor(rep("s0",times = length(unlist(Diff.F1.score.bag.tra.s0)))),
                            as.factor(rep("s3",times = length(unlist(Diff.F1.score.bag.tra.s3)))))
                                            )

Box.diff.F1.score.training.proc1 <- ggplot(boxplot.Diff.F1.score.bag.tra, 
                                  aes(x=training.set, y =Diff.F1.score.proc1, fill=var.selec)) + 
                                   geom_boxplot(alpha=0.4) +       # Fill transparency
                                   geom_hline(yintercept=0, color="red") + 
                                   theme_minimal() + scale_fill_brewer(palette="Accent") + 
                                   labs(x="training set", y="Increment in F1-score using c-rule") 


## Accuracy
Diff.Accuracy.bag.tra.s0<-list()
Diff.Accuracy.bag.tra.s3<-list()
for (j in 1:10)
{Diff.Accuracy.bag.tra.s0[[j]]<-Accuracy.bag.tra.c.opt.s0[[j]]-Accuracy.fitted.bag.c.1.s0[j]
Diff.Accuracy.bag.tra.s3[[j]]<-Accuracy.bag.tra.c.opt.s3[[j]]-Accuracy.fitted.bag.c.1.s3[j]}

boxplot.Diff.Accuracy.bag.tra <- data.frame(Diff.Accuracy.proc1 = c(unlist(Diff.Accuracy.bag.tra.s0),
                                                                    unlist(Diff.Accuracy.bag.tra.s3)), 
                                            training.set = c(as.factor(rep(1:10,times = sapply(Diff.Accuracy.bag.tra.s0,length))), 
                                                             as.factor(rep(1:10,times = sapply(Diff.Accuracy.bag.tra.s3,length)))), 
                                            var.selec = c(as.factor(rep("s0",times = length(unlist(Diff.Accuracy.bag.tra.s0)))),
                                                          as.factor(rep("s3",times = length(unlist(Diff.Accuracy.bag.tra.s3)))))
)

Box.diff.Accuracy.training.proc1 <- ggplot(boxplot.Diff.Accuracy.bag.tra, 
                                           aes(x=training.set, y =Diff.Accuracy.proc1, fill=var.selec)) + 
  geom_boxplot(alpha=0.4) +       # Fill transparency
  geom_hline(yintercept=0, color="red") + 
  theme_minimal() + scale_fill_brewer(palette="Accent") + 
  labs(x="training set", y="Increment in Accuracy using c-rule") 


## BA
Diff.BA.bag.tra.s0<-list()
Diff.BA.bag.tra.s3<-list()
for (j in 1:10)
{Diff.BA.bag.tra.s0[[j]]<-BA.bag.tra.c.opt.s0[[j]]-BA.fitted.bag.c.1.s0[j]
Diff.BA.bag.tra.s3[[j]]<-BA.bag.tra.c.opt.s3[[j]]-BA.fitted.bag.c.1.s3[j]}

boxplot.Diff.BA.bag.tra <- data.frame(Diff.BA.proc1 = c(unlist(Diff.BA.bag.tra.s0),
                                                                    unlist(Diff.BA.bag.tra.s3)), 
                                            training.set = c(as.factor(rep(1:10,times = sapply(Diff.BA.bag.tra.s0,length))), 
                                                             as.factor(rep(1:10,times = sapply(Diff.BA.bag.tra.s3,length)))), 
                                            var.selec = c(as.factor(rep("s0",times = length(unlist(Diff.BA.bag.tra.s0)))),
                                                          as.factor(rep("s3",times = length(unlist(Diff.BA.bag.tra.s3)))))
)

Box.diff.BA.training.proc1 <- ggplot(boxplot.Diff.BA.bag.tra, 
                                           aes(x=training.set, y =Diff.BA.proc1, fill=var.selec)) + 
  geom_boxplot(alpha=0.4) +       # Fill transparency
  geom_hline(yintercept=0, color="red") + 
  theme_minimal() + scale_fill_brewer(palette="Accent") + 
  labs(x="training set", y="Increment in BA using c-rule") 


