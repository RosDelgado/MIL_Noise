################################################################################
#######.  EXPERIMENTATION FOR SECTION 4.3 of the paper:
#######. 
#######.  A Simple Approach to Multiple Instance Learning:
#######.  Controlling the False Positive Rate
#######.     
#######.  by Rosario Delgado
#######.  December, 2024
################################################################################


# Cleaning environment
rm(list=ls());gc()

################################################################################
####### The Colon Cancer dataset can be downloaded from 
####### https://drive.google.com/file/d/1RcNlwg0TwaZoaFO0uMXHFtAo_DCVPE6z/view
####### (credit to Dr. Jiawen Yao)

####### We make preprocessing of images and predictions with the CausalMIL algorithm, which 
####### can be accessed to at https://github.com/WeijiaZhang24/CausalMIL
####### The predictions at the instance-level obtained by applying the CausalMIL algorithm to Colon Cancer 
####### dataset were saved into file: "expanded_results_2.csv". 

###### We load this CSV file and apply standard (c=1) and c-rule to the assigned labels
###### to the instances, in order to obtain the assigned labels for the bags.
###### Then, compare Accuracy, Balanced Accuracy and F1-score for the two procedures, 
###### based on a 5-fold cross-validation. 
###### 
################################################################################

library(ggplot2)

source("Algorithms.R")
source("mat_square.R")

F1.score<-function(A) # square 2 x 2 matrix. Classes 0, 1. Focused on class 0 
          {F<-2*A[1,1]/(2*A[1,1]+A[1,2]+A[2,1])
           return(F)}

Accuracy<-function(A) # square 2 x 2 matrix. 
         {Acc<-(A[1,1]+A[2,2])/sum(A)
         return(Acc)}

BA<-function(A) # square 2 x 2 matrix. 
    {Balance.Accuracy<-(A[1,1]/(A[1,1]+A[1,2])+A[2,2]/(A[2,2]+A[2,1]))/2
    return(Balance.Accuracy)}

################################################################################
### loading results from the CausalMIL algorithm applied to Colon Cancer dataset
### and mining the data
################################################################################

data.colon <- read.csv("expanded_results_2.csv")
View(data.colon)
table(data.colon$fold)  # total number of patches or instances by fold
table(data.colon$img)   # total number of patches or instances 
length(unique(data.colon$img))  # number of bags
length(unique(data.colon$img[which(data.colon$bag==0)]))    # number of bags "0"
length(unique(data.colon$img[which(data.colon$bag==1)]))    # number of bags "1"


# Change number of folds from 0,...,4 to 1,...,5
data.colon$fold<-data.colon$fold+1
table(data.colon$fold)  # total number of patches or instances by fold

length(unique(data.colon$img[which(data.colon$bag==0 & data.colon$fold==1)])) # number of bags "0" in fold 1
length(unique(data.colon$img[which(data.colon$bag==1 & data.colon$fold==1)])) # number of bags "1" in fold 1

length(unique(data.colon$img[which(data.colon$bag==0 & data.colon$fold==2)])) # number of bags "0" in fold 2
length(unique(data.colon$img[which(data.colon$bag==1 & data.colon$fold==2)])) # number of bags "1" in fold 2

length(unique(data.colon$img[which(data.colon$bag==0 & data.colon$fold==3)])) # number of bags "0" in fold 3
length(unique(data.colon$img[which(data.colon$bag==1 & data.colon$fold==3)])) # number of bags "1" in fold 3

length(unique(data.colon$img[which(data.colon$bag==0 & data.colon$fold==4)])) # number of bags "0" in fold 4
length(unique(data.colon$img[which(data.colon$bag==1 & data.colon$fold==4)])) # number of bags "1" in fold 4

length(unique(data.colon$img[which(data.colon$bag==0 & data.colon$fold==5)])) # number of bags "0" in fold 5
length(unique(data.colon$img[which(data.colon$bag==1 & data.colon$fold==5)])) # number of bags "1" in fold 5


max(data.colon$prediction)  # 0.8317857
max(data.colon$prediction[which(data.colon$fold==1)]) # 0.8317857
max(data.colon$prediction[which(data.colon$fold==2)]) # 0.7895024
max(data.colon$prediction[which(data.colon$fold==3)]) # 0.7364969
max(data.colon$prediction[which(data.colon$fold==4)]) # 0.7940918
max(data.colon$prediction[which(data.colon$fold==5)]) # 0.7720776


min(data.colon$prediction)  # 5.73e-05
min(data.colon$prediction[which(data.colon$fold==1)]) # 6.88e-05
min(data.colon$prediction[which(data.colon$fold==2)]) # 0.000239352
min(data.colon$prediction[which(data.colon$fold==3)]) # 0.000327245
min(data.colon$prediction[which(data.colon$fold==4)]) # 0.00019886
min(data.colon$prediction[which(data.colon$fold==5)]) # 5.73e-05

thres<-seq(from=0.01, to=0.50, by=0.01)  # 41 diferent thresholds


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
####   STEP 1: PREDICTING INSTANCE LABELS from "prediction"
####           BASED ON DIFFERENT THRESHOLDS (in "thres"): pred.label.inst[[t]]
################################################################################
#
pred.label.inst<-list()

for (t in 1:length(thres))  # for each threshold, different instance-level labels 
{
  pred.label.inst[[t]]<-vector()
  for (i in 1:dim(data.colon)[1])
    {pred.label.inst[[t]][i]<-ifelse(data.colon$prediction[i]>thres[t],1,0)
  }
}

lapply(pred.label.inst,sum)

################################################################################
####   STEP 2: ESTIMATING p = false negative rate (instance level)
####           ESTIMATING q = false positive rate (instance level) 
####           DEPENDING ON THE THRESHOLD, FOR ANY FOLD
################################################################################
#

tables.instances<-list() # confusion matrices for instances

est.p<-list()
est.q<-list()

ff<-list()   # instance numbers at each fold

for (i in 1:5) # for each fold
{ tables.instances[[i]]<-list()
  est.p[[i]]<-vector()
  est.q[[i]]<-vector()
  ff[[i]]<-which(data.colon$fold==i)
  for (t in 1:length(thres))
  { 
   tables.instances[[i]][[t]]<-table(pred.label.inst[[t]][ff[[i]]],data.colon$actual[ff[[i]]])
   est.p[[i]][t]<-tables.instances[[i]][[t]][1,2]/sum(tables.instances[[i]][[t]])
   est.q[[i]][t]<-tables.instances[[i]][[t]][2,1]/sum(tables.instances[[i]][[t]])
  }
}  


################################################################################
#########
######### STEP 3: Decision about the bags in any fold using standard procedure (c=1)
#########         from the predictions in pred.label.inst[[t]]
########
################################################################################

bag.name<-list()  # bags' names in the folds
number.of.bags<-vector()

for (i in 1:5)
{bag.name[[i]]<-unique(data.colon$img[ff[[i]]])
number.of.bags[i]<-length(bag.name[[i]])}

#
##
#

size.bags<-list()   # bag's sizes, for each bag in each fold

for (i in 1:5)
{size.bags[[i]]<-vector()
  for (j in 1:number.of.bags[i])
  {size.bags[[i]][j]<-length(which(data.colon$img==bag.name[[i]][j]))
  }
}

unlist(lapply(size.bags,sum)) # verification

#
##
#

obs.bag<-list()  # bag's observed labels 
for (i in 1:5)
{obs.bag[[i]]<-vector()
 for (j in 1:number.of.bags[i])
   {obs.bag[[i]][j]<-unique(data.colon$bag[which(data.colon$img==bag.name[[i]][j])])
   }
}

#
##
#

pred.bag<-list()  # bag's predicted labels with standard c=1

for (i in 1:5)  # for any fold
{pred.bag[[i]]<-list()
for (t in 1:length(thres)) # for any threshold
  {pred.bag[[i]][[t]]<-vector()
  for (j in 1:number.of.bags[i])  # for any bag in the fold 
     {pred.bag[[i]][[t]][j]<-if_else(sum(pred.label.inst[[t]][which(data.colon$img==bag.name[[i]][j])])>0,1,0)
     }
   }
}

#
##
#

table.bag.c.1<-list()    # bag's confusion matrices with standard c=1, 
                         # for any fold i, and threshold t     
for (i in 1:5)   # for any fold
 {table.bag.c.1[[i]]<-list()
 for (t in 1:length(thres)) # for any threshold
   {table.bag.c.1[[i]][[t]]<-table(pred.bag[[i]][[t]],obs.bag[[i]])
   }
 }

 
   ################################################################################
   ###### Metrics for predicted bag labels with standard c=1
   ################################################################################

F1.score.bag.c.1<-list()

for (t in 1:length(thres))
{F1.score.bag.c.1[[t]]<-vector()
 for (i in 1:5)
   {F1.score.bag.c.1[[t]][i]<-F1.score(mat.square(table.bag.c.1[[i]][[t]],c("0","1")))
 }
}

F1.score.bag.c.1


##

Accuracy.bag.c.1<-list()

for (t in 1:length(thres))
{Accuracy.bag.c.1[[t]]<-vector()
for (i in 1:5)
{Accuracy.bag.c.1[[t]][i]<-Accuracy(mat.square(table.bag.c.1[[i]][[t]],c("0","1")))
}
}

Accuracy.bag.c.1

##

BA.bag.c.1<-list()

for (t in 1:length(thres))
{BA.bag.c.1[[t]]<-vector()
for (i in 1:5)
{BA.bag.c.1[[t]][i]<-BA(mat.square(table.bag.c.1[[i]][[t]],c("0","1")))
}
}

BA.bag.c.1



################################################################################
#########
######### STEP 4: Decision about the bags in any fold using new procedure (c optimal, c*)
#########         from the predictions in pred.label.inst[[t]]
########
################################################################################
##
## We use this approach as if we did not know the actual labels of the instances!!

pi.estimated<-list()  # estimated pi = proportion of positively classified instances 
                      # in each bag of each fold

for (i in 1:5)
{pi.estimated[[i]]<-list()
 for (t in 1:length(thres))
 {pi.estimated[[i]][[t]]<-vector()
  for (j in 1:number.of.bags[i])   
    {pi.estimated[[i]][[t]][j]<-sum(pred.label.inst[[t]][which(data.colon$img==bag.name[[i]][j])])/
                                length(pred.label.inst[[t]][which(data.colon$img==bag.name[[i]][j])])
    }
 }
}

#########

alpha<-0.10
beta<-0.10

#########

m.plus<-list()
m.minus<-list()

for (i in 1:5)
{m.plus[[i]]<-list()
 m.minus[[i]]<-list()
 for (t in 1:length(thres))
  {m.plus[[i]][[t]]<-vector()
   m.minus[[i]][[t]]<-vector()
   for (j in 1:number.of.bags[i])   
    {m.plus[[i]][[t]][j]<-as.numeric(m.plus.minus(size.bags[[i]][j],
                                                   est.p[[i]][t],est.q[[i]][t],
                                                   pi.estimated[[i]][[t]][j])[[2]][1])
    m.minus[[i]][[t]][j]<-as.numeric(m.plus.minus(size.bags[[i]][j],
                                                 est.p[[i]][t],est.q[[i]][t],
                                                 pi.estimated[[i]][[t]][j])[[2]][2])
    }
  }
}



#######

result.opt.c<-list()

for (i in 1:5)   # for any fold
{result.opt.c[[i]]<-list()
 for (t in 1:length(thres)) # for any threshold
   {result.opt.c[[i]][[t]]<-list()
    for (j in 1:number.of.bags[i])  # for any bag in the fold 
     {result.opt.c[[i]][[t]][[j]]<-optimal.c(size.bags[[i]][j],
                                est.p[[i]][t],est.q[[i]][t],
                                alpha,beta,
                                m.plus[[i]][[t]][j],m.minus[[i]][[t]][j])
     }
 }
}


#########
### Decision about the bags with c-rule, c given by result.opt.c


pred.bag.c.opt<-list()   

for (i in 1:5)    # for any fols
 {pred.bag.c.opt[[i]]<-list() 
  for (t in 1:length(thres))     # for any threshold
   {pred.bag.c.opt[[i]][[t]]<-vector()
    for (j in 1:number.of.bags[i])  # for any bag in the fold
     {if (is.null(result.opt.c[[i]][[t]][[j]][5,2])==TRUE)
       {pred.bag.c.opt[[i]][[t]][j]<-NA} else {
       if(is.na(as.numeric(result.opt.c[[i]][[t]][[j]][5,2]))==FALSE)
         {pred.bag.c.opt[[i]][[t]][j]<-if_else(sum(pred.label.inst[[t]][which(data.colon$img==bag.name[[i]][j])]) >
                as.numeric(result.opt.c[[i]][[t]][[j]][5,2])-1,1,0)
         } else {pred.bag.c.opt[[i]][[t]][j]<-NA}
       }
     }
  }
}
  

#
##
#

table.bag.c.opt<-list()    # bag's confusion matrices with c*, 
# for any fold i, and threshold t     
for (i in 1:5)   # for any fold
{table.bag.c.opt[[i]]<-list()
for (t in 1:length(thres)) # for any threshold
{table.bag.c.opt[[i]][[t]]<-table(pred.bag.c.opt[[i]][[t]],obs.bag[[i]])
}
}


################################################################################
###### Metrics for predicted bag labels with c*
################################################################################

F1.score.bag.c.opt<-list()

for (t in 1:length(thres))
{F1.score.bag.c.opt[[t]]<-vector()
for (i in 1:5)
{F1.score.bag.c.opt[[t]][i]<-F1.score(mat.square(table.bag.c.opt[[i]][[t]],c("0","1")))
}
}

F1.score.bag.c.opt

##

Accuracy.bag.c.opt<-list()

for (t in 1:length(thres))
{Accuracy.bag.c.opt[[t]]<-vector()
for (i in 1:5)
{Accuracy.bag.c.opt[[t]][i]<-Accuracy(mat.square(table.bag.c.opt[[i]][[t]],c("0","1")))
}
}

Accuracy.bag.c.opt

##

BA.bag.c.opt<-list()

for (t in 1:length(thres))
{BA.bag.c.opt[[t]]<-vector()
for (i in 1:5)
{BA.bag.c.opt[[t]][i]<-BA(mat.square(table.bag.c.opt[[i]][[t]],c("0","1")))
}
}

BA.bag.c.opt



################################################################################
#########
######### STEP 5: Metrics comparison for the predicted bag labels with new procedure c optimal (c*),
#########          and predicted bag labels with c=1, from the predicted instance labels 
#########         nobtained with different thresholds
################################################################################
##
################################################################################
###### 
################################################################################

p.val.F1.c.opt<-vector()
p.val.F1.c.1<-vector()

for (t in 1:length(thres))
{ if (sum(!is.na(F1.score.bag.c.opt[[t]]-F1.score.bag.c.1[[t]])) >= 3)
 {p.sw<-shapiro.test(F1.score.bag.c.opt[[t]]-F1.score.bag.c.1[[t]])$p.value
 if (p.sw<0.05)
 {p.val.F1.c.opt[t]<-wilcox.test(F1.score.bag.c.opt[[t]],F1.score.bag.c.1[[t]],paired=TRUE,alternative="greater")$p.value
 p.val.F1.c.1[t]<-wilcox.test(F1.score.bag.c.opt[[t]],F1.score.bag.c.1[[t]],paired=TRUE,alternative="less")$p.value}
 else
 {p.val.F1.c.opt[t]<-t.test(F1.score.bag.c.opt[[t]],F1.score.bag.c.1[[t]],paired=TRUE,alternative="greater")$p.value
 p.val.F1.c.1[t]<-t.test(F1.score.bag.c.opt[[t]],F1.score.bag.c.1[[t]],paired=TRUE,alternative="less")$p.value}
} else 
{p.val.F1.c.opt[t]<-NA
p.val.F1.c.1[t]<-NA}
}

p.val.F1.c.opt
p.val.F1.c.1

sum(p.val.F1.c.opt<0.05)
sum(p.val.F1.c.1<0.05)


##


p.val.Acc.c.opt<-vector()
p.val.Acc.c.1<-vector()

for (t in 1:length(thres))
{ if (sum(!is.na(Accuracy.bag.c.opt[[t]]-Accuracy.bag.c.1[[t]])) >= 3)
  {p.sw<-shapiro.test(Accuracy.bag.c.opt[[t]]-Accuracy.bag.c.1[[t]])$p.value
  if (p.sw<0.05)
  {p.val.Acc.c.opt[t]<-wilcox.test(Accuracy.bag.c.opt[[t]],Accuracy.bag.c.1[[t]],paired=TRUE,alternative="greater")$p.value
  p.val.Acc.c.1[t]<-wilcox.test(Accuracy.bag.c.opt[[t]],Accuracy.bag.c.1[[t]],paired=TRUE,alternative="less")$p.value}
  else
  {p.val.Acc.c.opt[t]<-t.test(Accuracy.bag.c.opt[[t]],Accuracy.bag.c.1[[t]],paired=TRUE,alternative="greater")$p.value
  p.val.Acc.c.1[t]<-t.test(Accuracy.bag.c.opt[[t]],Accuracy.bag.c.1[[t]],paired=TRUE,alternative="less")$p.value}
} else
{p.val.Acc.c.opt[t]<-NA
p.val.Acc.c.1[t]<-NA}
}


p.val.Acc.c.opt
p.val.Acc.c.1

sum(p.val.Acc.c.opt<0.05)
sum(p.val.Acc.c.1<0.05)

##


p.val.BA.c.opt<-vector()
p.val.BA.c.1<-vector()

for (t in 1:length(thres))
{ if (sum(!is.na(BA.bag.c.opt[[t]]-BA.bag.c.1[[t]])) >= 3)
  {p.sw<-shapiro.test(BA.bag.c.opt[[t]]-BA.bag.c.1[[t]])$p.value
  if (p.sw<0.05)
  {p.val.BA.c.opt[t]<-wilcox.test(BA.bag.c.opt[[t]],BA.bag.c.1[[t]],paired=TRUE,alternative="greater")$p.value
  p.val.BA.c.1[t]<-wilcox.test(BA.bag.c.opt[[t]],BA.bag.c.1[[t]],paired=TRUE,alternative="less")$p.value}
  else
  {p.val.BA.c.opt[t]<-t.test(BA.bag.c.opt[[t]],BA.bag.c.1[[t]],paired=TRUE,alternative="greater")$p.value
  p.val.BA.c.1[t]<-t.test(BA.bag.c.opt[[t]],BA.bag.c.1[[t]],paired=TRUE,alternative="less")$p.value}
} else 
{p.val.BA.c.opt[t]<-NA
p.val.BA.c.1[t]<-NA}
}

p.val.BA.c.opt
p.val.BA.c.1

sum(p.val.BA.c.opt<0.05,na.rm=TRUE)
sum(p.val.BA.c.1<0.05,na.rm=TRUE)


##########################################################
##########################################################
### 
### Now we make some boxplots for the predictions of the bags with the new procedure
### c.opt (c*) and the standard (c=1),
### for the results obtained for any threshold, for any of the 5 folds
### 
###
###########################################################
###########################################################



## F1.score
Diff.F1.score.bag<-list()

for (i in 1:5)
{Diff.F1.score.bag[[i]]<-vector()
 for (t in 1:length(thres))
 {Diff.F1.score.bag[[i]][t]<-F1.score.bag.c.opt[[t]][i]-F1.score.bag.c.1[[t]][i]
 }
}


boxplot.Diff.F1.score.bag <- data.frame(Diff.F1.score = unlist(Diff.F1.score.bag), 
                fold = as.factor(rep(1:5,times = sapply(Diff.F1.score.bag,length)))
                                            )

Box.diff.F1.score <- ggplot(boxplot.Diff.F1.score.bag, 
                                  aes(x=fold, y =Diff.F1.score)) + 
                                   geom_boxplot(alpha=0.4) +       # Fill transparency
                                   geom_hline(yintercept=0, color="red") + 
                                   theme_minimal() + scale_fill_brewer(palette="Accent") + 
                                   ggtitle("Increment in F1-score using c-rule with respect to c=1")+
                                   labs(x="fold", y=" ") 

## Accuracy
Diff.Accuracy.bag<-list()

for (i in 1:5)
{Diff.Accuracy.bag[[i]]<-vector()
for (t in 1:length(thres))
{Diff.Accuracy.bag[[i]][t]<-Accuracy.bag.c.opt[[t]][i]-Accuracy.bag.c.1[[t]][i]
}
}


boxplot.Diff.Accuracy.bag <- data.frame(Diff.Accuracy = unlist(Diff.Accuracy.bag), 
                                        fold = as.factor(rep(1:5,times = sapply(Diff.Accuracy.bag,length)))
)

Box.diff.Accuracy <- ggplot(boxplot.Diff.Accuracy.bag, 
                            aes(x=fold, y =Diff.Accuracy)) + 
  geom_boxplot(alpha=0.4) +       # Fill transparency
  geom_hline(yintercept=0, color="red") + 
  theme_minimal() + scale_fill_brewer(palette="Accent") + 
  ggtitle("Increment in Accuracy using c-rule with respect to c=1")+
  labs(x="fold", y=" ") 


## BA
Diff.BA.bag<-list()

for (i in 1:5)
{Diff.BA.bag[[i]]<-vector()
for (t in 1:length(thres))
{Diff.BA.bag[[i]][t]<-BA.bag.c.opt[[t]][i]-BA.bag.c.1[[t]][i]
}
}


boxplot.Diff.BA.bag <- data.frame(Diff.BA = unlist(Diff.BA.bag), 
                                        fold = as.factor(rep(1:5,times = sapply(Diff.BA.bag,length)))
)

Box.diff.BA <- ggplot(boxplot.Diff.BA.bag, 
                            aes(x=fold, y =Diff.BA)) + 
  geom_boxplot(alpha=0.4) +       # Fill transparency
  geom_hline(yintercept=0, color="red") + 
  theme_minimal() + scale_fill_brewer(palette="Accent") + 
  ggtitle("Increment in BA using c-rule with respect to c=1")+
  labs(x="fold", y=" ") 



###
Box.diff.F1.score
Box.diff.Accuracy
Box.diff.BA
###


############################
#####.   PLOTS
############################

###. F1-score


mean.Diff.F1.score.bag.folds<-vector()
for (t in 1:length(thres))
{mean.Diff.F1.score.bag.folds[t]<-mean(c(Diff.F1.score.bag[[1]][t],
                                     Diff.F1.score.bag[[2]][t],
                                     Diff.F1.score.bag[[3]][t],
                                     Diff.F1.score.bag[[4]][t],
                                     Diff.F1.score.bag[[5]][t]
                                     ))}



mean.except.3.Diff.F1.score.bag.folds<-vector()
for (t in 1:length(thres))
{mean.except.3.Diff.F1.score.bag.folds[t]<-mean(c(Diff.F1.score.bag[[1]][t],
                                         Diff.F1.score.bag[[2]][t],
                                         Diff.F1.score.bag[[4]][t],
                                         Diff.F1.score.bag[[5]][t]
))}


plot(
  thres, 
  mean.Diff.F1.score.bag.folds, 
  type = "l", 
  col = "blue",             
  lwd = 1,                  
  xlab = "Threshold",      
  ylab = " ", 
  main = "Mean increment in F1-score using c-rule across the folds"  
)

abline(h = 0, col = "red", lty = 2, lwd = 2)



plot(
  thres,
  mean.except.3.Diff.F1.score.bag.folds,
  type = "l",
  col = "blue",             
  lwd = 1,                 
  xlab = "Threshold",       
  ylab = " ", 
  main = "Mean increment in F1-score using c-rule (except fold 3)"  
)

abline(h = 0, col = "red", lty = 2, lwd = 2)

#
##
#

df.F1.score <- do.call(rbind, lapply(1:5, function(i) {
  data.frame(thres = thres, value = Diff.F1.score.bag[[i]], fold = paste0("Fold ", i))
}))

# 

fold_colors <- c("Fold 1" = "#1b9e77", 
                 "Fold 2" = "#d95f02", 
                 "Fold 3" = "#7570b3", 
                 "Fold 4" = "#e7298a", 
                 "Fold 5" = "#66a61e")


ggplot(df.F1.score, aes(x = thres, y = value, color = fold)) +
  geom_line(size = 0.4) +                           
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +  
  labs(x = "Threshold", y = " ", color = "Fold", title = "Increment in F1-score using c-rule with respect to c=1") +
  theme_minimal()+ 
  scale_color_manual(values = fold_colors) +
theme(legend.position = "bottom")

###. Accuracy


mean.Diff.Accuracy.bag.folds<-vector()
for (t in 1:length(thres))
{mean.Diff.Accuracy.bag.folds[t]<-mean(c(Diff.Accuracy.bag[[1]][t],
                                         Diff.Accuracy.bag[[2]][t],
                                         Diff.Accuracy.bag[[3]][t],
                                         Diff.Accuracy.bag[[4]][t],
                                         Diff.Accuracy.bag[[5]][t]
))}



mean.except.3.Diff.Accuracy.bag.folds<-vector()
for (t in 1:length(thres))
{mean.except.3.Diff.Accuracy.bag.folds[t]<-mean(c(Diff.Accuracy.bag[[1]][t],
                                         Diff.Accuracy.bag[[2]][t],
                                         Diff.Accuracy.bag[[4]][t],
                                         Diff.Accuracy.bag[[5]][t]
))}


plot(
  thres, 
  mean.Diff.Accuracy.bag.folds, 
  type = "l", 
  col = "blue",             
  lwd = 1,                 
  xlab = "Threshold",      
  ylab = " ",  
  main = "Mean increment in Accuracy using c-rule across the folds"  
)

abline(h = 0, col = "red", lty = 2, lwd = 2)



plot(
  thres,
  mean.except.3.Diff.Accuracy.bag.folds,
  type = "l",
  col = "blue",            
  lwd = 1,                  
  xlab = "Threshold",       
  ylab = " ", 
  main = "Mean increment in Accuracy using c-rule (except fold 3)"  
)

abline(h = 0, col = "red", lty = 2, lwd = 2)

#
##
#

df.Accuracy <- do.call(rbind, lapply(1:5, function(i) {
  data.frame(thres = thres, value = Diff.Accuracy.bag[[i]], fold = paste0("Fold ", i))
}))

# 
ggplot(df.Accuracy, aes(x = thres, y = value, color = fold)) +
  geom_line(size = 0.4) +                           
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +  
  labs(x = "Threshold", y = " ", color = "Fold", title = "Increment in Accuracy using c-rule with respect to c=1") +
  theme_minimal()+ 
  scale_color_manual(values = fold_colors) +
   theme(legend.position = "bottom")                             



###. BA


mean.Diff.BA.bag.folds<-vector()
for (t in 1:length(thres))
{mean.Diff.BA.bag.folds[t]<-mean(c(Diff.BA.bag[[1]][t],
                                         Diff.BA.bag[[2]][t],
                                         Diff.BA.bag[[3]][t],
                                         Diff.BA.bag[[4]][t],
                                         Diff.BA.bag[[5]][t]
))}



mean.except.3.Diff.BA.bag.folds<-vector()
for (t in 1:length(thres))
{mean.except.3.Diff.BA.bag.folds[t]<-mean(c(Diff.BA.bag[[1]][t],
                                         Diff.BA.bag[[2]][t],
                                         Diff.BA.bag[[4]][t],
                                         Diff.BA.bag[[5]][t]
))}


plot(
  thres, 
  mean.Diff.BA.bag.folds, 
  type = "l", 
  col = "blue",             
  lwd = 1,                 
  xlab = "Threshold",       
  ylab = " ", 
  main = "Mean increment in BA using c-rule across the folds"  
)

abline(h = 0, col = "red", lty = 2, lwd = 2)



plot(
  thres,
  mean.except.3.Diff.BA.bag.folds,
  type = "l",
  col = "blue",             
  lwd = 1,                  
  xlab = "Threshold",       
  ylab = " ",  
  main = "Mean increment in BA using c-rule (except fold 3)"  
)

abline(h = 0, col = "red", lty = 2, lwd = 2)

#
##
#


df.BA <- do.call(rbind, lapply(1:5, function(i) {
  data.frame(thres = thres, value = Diff.BA.bag[[i]], fold = paste0("Fold ", i))
}))

# 
ggplot(df.BA, aes(x = thres, y = value, color = fold)) +
  geom_line(size = 0.5) +                           
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +  
  labs(x = "Threshold", y = " ", color = "Fold", title = "Increment in BA using c-rule with respect to c=1") +
  theme_minimal()+ 
  scale_color_manual(values = fold_colors) +
  theme(legend.position = "bottom")                                   


#############################################
#############################################
###. PLOT of p.est and q.est
#############################################

df.p.est<-do.call(rbind,lapply(1:5,function(i){
  data.frame(thres = thres, value = est.p[[i]], fold = paste0("Fold ", i)) 
  }))

df.q.est<-do.call(rbind,lapply(1:5,function(i){
  data.frame(thres = thres, value = est.q[[i]], fold = paste0("Fold ", i)) 
}))


df.p.est$group <- "p"
df.q.est$group <- "q"
df.p.q.est <- rbind(df.p.est, df.q.est)



fold_colors <- c("Fold 1" = "#1b9e77", 
                 "Fold 2" = "#d95f02", 
                 "Fold 3" = "#7570b3", 
                 "Fold 4" = "#e7298a", 
                 "Fold 5" = "#66a61e")

# Gráfico para p
plot_p <- ggplot(df.p.est, aes(x = thres, y = value, color = fold)) +
  geom_line(size = 0.5) +
  labs(x = "Threshold", 
       y = "Estimated p", 
       color = "Fold") +
  scale_color_manual(values = fold_colors) +
  theme_minimal()+ 
  theme(legend.position = "bottom", legend.box="vertical")

# Gráfico para q
plot_q <- ggplot(df.q.est, aes(x = thres, y = value, color = fold)) +
  geom_line(size = 0.5)+
  #, linetype = "dashed") +
  labs(x = "Threshold", 
       y = "Estimated q", 
       color = "Fold") +
  scale_color_manual(values = fold_colors) +
  theme_minimal() + 
theme(legend.position = "none")

# Combining plots  
combined_plot <- plot_p + plot_q + 
  plot_layout(guides = "collect") +
  plot_annotation(title = " ",
                  theme = theme(plot.title = element_text(hjust = 0.5))) &
   theme(legend.position = "bottom", 
         legend.box = "vertical", 
         legend.text = element_text(size = 10),
         legend.title = element_text(size = 12, face = "bold"))

print(combined_plot)





