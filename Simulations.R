

#######.  SIMULATIONS FOR SECTION 4 
#######.  A Simple Approach to Multiple Instance Learning with Noisy Instance Labels
#######.  by Rosario Delgado
#######.  May, 2024



# Cleaning environment
rm(list=ls());gc()


library(ggplot2)
source("Algorithms.R")

##############################
### scenario 1

M<-100


n<-M*0.1


v.p<-seq(from = 0.001, to = 0.499, by=0.005)  # 100 values
v.q<-seq(from = 0.001, to = 0.499, by=0.005)  # 100 values

alpha<-0.05
beta<-0.10

values.p.plus<-seq(from=0.001, to=0.01, by=0.001)   # scarce positive

#####



vector.p.q<-data.frame(p = v.p, q = rep(v.q,each=length(v.p)))  # 10 000 combinations

##########



set.seed(1234)
set.seed.vector<-sample(1:10000,length(values.p.plus),replace=FALSE)

M.simulated<-list()
for (i in 1:length(values.p.plus))
{set.seed(set.seed.vector[i])
  M.simulated[[i]]<-sample(c(1,0),M,replace=TRUE, prob=c(values.p.plus[i],1-values.p.plus[i]))
}

#########
### Correct decision about M.simulated


correct.decision<-vector()
for (i in 1:length(M.simulated))
{ if (sum(M.simulated[[i]])==0)
{correct.decision[i]<-0} else {correct.decision[i]<-1}
}


###########  CONSIDERING ERRORS IN CLASSIFICATION p AND q
success.decision.c.opt<-vector()
success.standard.decision.mod<-vector()

Delta.success<-vector()


for (k in 1:dim(vector.p.q)[1])
{
  
  
  
  set.seed(k)
  set.seed.vector.2<-sample(1:(M^3),M,replace=FALSE)
  
  
  M.simulated.noisy<-list()
  for (i in 1:length(M.simulated))
  {M.simulated.noisy[[i]]<-M.simulated[[i]]
  for (j in 1:length(M.simulated[[i]]))
  {if (M.simulated[[i]][j]==1)
  {set.seed(set.seed.vector.2[j])
    M.simulated.noisy[[i]][j]<-sample(c(0,1),1,prob=c(vector.p.q[k,1],1-vector.p.q[k,1]))} else {set.seed(set.seed.vector.2[j])
      M.simulated.noisy[[i]][j]<-sample(c(1,0),1,prob=c(vector.p.q[k,2],1-vector.p.q[k,2]))}
  }
  }
  
  
  
  
  ########
  n.simulated.noisy<-list()
  for (i in 1:length(M.simulated.noisy))
  {set.seed(set.seed.vector.2[i])
    n.simulated.noisy[[i]]<-sample(M.simulated.noisy[[i]],n,replace=FALSE)
  }
  
  pi.simulated<-lapply(n.simulated.noisy,mean)
  
  #########
  
  
  m.plus.simulated<-vector()
  m.minus.simulated<-vector()
  for (i in 1:length(pi.simulated))
  {m.plus.simulated[i]<-as.numeric(m.plus.minus(M,vector.p.q[k,1],vector.p.q[k,2],pi.simulated[[i]])[[2]][1])
  m.minus.simulated[i]<-as.numeric(m.plus.minus(M,vector.p.q[k,1],vector.p.q[k,2],pi.simulated[[i]])[[2]][2])
  }
  
  
  #######
  
  result.opt.c<-list()
  for (i in 1:length(pi.simulated))
  {result.opt.c[[i]]<-optimal.c(M,vector.p.q[k,1],vector.p.q[k,2],alpha,beta,m.plus.simulated[i],m.minus.simulated[i])
  }
  
  
  #########
  ### Decision about M.simulated.noisy with standard procedure (c=1)
  
  standard.decision.c.1<-vector()    # 1 = +, 0 = -
  for (i in 1:length(M.simulated.noisy))
  { if (sum(M.simulated.noisy[[i]])==0)
  {standard.decision.c.1[i]<-0} else {standard.decision.c.1[i]<-1}
  }
  
  
  error.standard.decision<-sum(correct.decision!=standard.decision.c.1,na.rm=TRUE)
  success.standard.decision<-sum(correct.decision==standard.decision.c.1,na.rm=TRUE)
  
  #########
  ### Decision about M.simulated.noisy with c-rule, c given by result.opt.c
  
  
  decision.c.opt<-vector()    # 1 = +, 0 = -
  for (i in 1:length(M.simulated.noisy))
  { if (is.na(as.numeric(result.opt.c[[i]][5,2]))==FALSE)
  { if(sum(M.simulated.noisy[[i]]) < as.numeric(result.opt.c[[i]][5,2]))
  {decision.c.opt[i]<-0} else {decision.c.opt[i]<-1}
  } else {decision.c.opt[i]<-NA}
  }
  
  error.decision.c.opt<-sum(correct.decision!=decision.c.opt,na.rm=TRUE)
  success.decision.c.opt[k]<-sum(correct.decision==decision.c.opt,na.rm=TRUE)
  
  no.na<-which(is.na(decision.c.opt)==FALSE)
  success.standard.decision.mod[k]<-sum(correct.decision[no.na]==standard.decision.c.1[no.na])
  
  #########  COMPARISON STANDARD (c=1) vs c-rule
  #error.standard.decision
  #error.decision.c.opt
  
  Delta.success[k]<-success.decision.c.opt[k]-success.standard.decision.mod[k]
  #Delta.success    # + indica millor c-rule
  # - indica millor standard rule
  # 0 indica igual
  print(k)
}   # END FOR k



length(Delta.success)
length(which(Delta.success<0))
length(which(Delta.success==0))
length(which(Delta.success>0))

vector.p.q[which(Delta.success<0),]
summary(vector.p.q[which(Delta.success<0),])
sd(vector.p.q[which(Delta.success<0),][[1]])
sd(vector.p.q[which(Delta.success<0),][[2]])

par(pty = "s")
boxplot(vector.p.q[which(Delta.success<0),],
        col = c("#003C67FF", "#EFC000FF"),
        main = "Scenario 1. Distribution of p and q for Delta<0")
#xlab = " ", ylab = " ")


par(pty = "s")
boxplot(vector.p.q,
        col = c("#003C67FF", "#EFC000FF"),
        main = "Scenario 1. Distribution of p and q")
#xlab = " ", ylab = " ")


prop.best.c.rule<-length(which(Delta.success>0))/(length(which(Delta.success!=0)))
prop.best.c.rule


length(success.decision.c.opt)
length(success.standard.decision.mod)

mean(success.decision.c.opt)
mean(success.standard.decision.mod)

summary(success.decision.c.opt)
summary(success.standard.decision.mod)
summary(Delta.success)

sd(success.decision.c.opt)
sd(success.standard.decision.mod)
sd(Delta.success)

cor.test(x = success.standard.decision.mod, y = success.decision.c.opt, 
         method = c("pearson"), 
         conf.level = 0.95)

result<-as.data.frame(cbind(success.standard.decision.mod,success.decision.c.opt))
colnames(result)<-c("standard", "c-rule")

par(pty = "s")
boxplot(result,
        col = c("#003C67FF", "#EFC000FF"),
        main = "Comparison of rules. Scenario 1",
        xlab = "rule", ylab = "Number of successes")

# test of equality of variances
bartlett.test(result)

t.test(x=result[[1]], y=result[[2]],
       alternative = "less",
       mu = 0, 
       paired = TRUE,   
       var.equal = FALSE,
       conf.level = 0.95)







###### HEATMAP OF Delta.success with respect to p and q

Delta.success.df<-as.data.frame(cbind(vector.p.q,Delta.success))
colnames(Delta.success.df)<-c("p","q","Increment")
Delta.success.df


min<-min(Delta.success.df[[3]])
Max<-max(Delta.success.df[[3]])
med<-median(Delta.success.df[[3]])

heat.map.Delta.success.M.100<-ggplot(data = Delta.success.df, aes(x=p, y=q, fill=Increment)) + 
  geom_tile()+
  ggtitle("Scenario 1. M = 100. alpha=0.05, beta=0.10") +
  scale_fill_gradient2(low = "#FF0033", high = "#6666FF", mid = "white", # #FF0033
                       midpoint = 0, limit = c(min,Max), space = "Lab", 
                       name="Increment") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))


print(heat.map.Delta.success.M.100)


##############################
##############################




##############################
### scenario 2: 

M<-seq(from=10, to=100, by=10)

n<-M*0.2

v.p<-seq(from = 0.001, to = 0.3, by=0.01)  # 30 values
v.q<-seq(from = 0.001, to = 0.3, by=0.01)  # 30 values

alpha<-0.05
beta<-0.10

positive.bag<-list()
for (j in 1:length(M))
  {positive.bag[[j]]<-c(1,rep(0,times=(M[j]-1)))
}

negative.bag<-list()
for (j in 1:length(M))
{negative.bag[[j]]<-rep(0,times=M[j])
}

#####

vector.p.q<-data.frame(p = v.p, q = rep(v.q,each=length(v.p)))  # 900 combinaciones de valores

##########

### For each value of M, we have 670 positive bags (identical) and 1040 negative bags.

###########  CONSIDERING ERRORS IN CLASSIFICATION p AND q
num.positive.bags<-670
num.negative.bags<-1040

positive.bag.noisy<-list()
negative.bag.noisy<-list()
n.simulated.positive.bag.noisy<-list()
n.simulated.negative.bag.noisy<-list()

pi.simulated.positive.bag<-list()
pi.simulated.negative.bag<-list()

m.plus.simulated.positive.bag<-list()
m.minus.simulated.positive.bag<-list()
m.plus.simulated.negative.bag<-list()
m.minus.simulated.negative.bag<-list()

result.opt.c.positive.bag<-list()
result.opt.c.negative.bag<-list()

standard.decision.c.1.positive.bag<-list()
standard.decision.c.1.negative.bag <-list()

error.standard.decision.positive.bag<-list()
success.standard.decision.positive.bag<-list()
error.standard.decision.negative.bag<-list()
success.standard.decision.negative.bag<-list()
success.standard.decision.positive.bag.mod<-list()
success.standard.decision.negative.bag.mod<-list()
success.standard.decision.all.mod<-list()

decision.c.opt.positive.bag<-list()
decision.c.opt.negative.bag <-list()

success.decision.c.opt.positive.bag <- list()
error.decision.c.opt.positive.bag <- list()
success.decision.c.opt.negative.bag <- list()
error.decision.c.opt.negative.bag <- list()
success.decision.c.opt.all<-list()

Delta.success.positive.bag <- list()
Delta.success.negative.bag <-list()

for (j in 1:length(M))
{

error.standard.decision.positive.bag[[j]]<-vector()
success.standard.decision.positive.bag[[j]]<-vector()
error.standard.decision.negative.bag[[j]]<-vector()
success.standard.decision.negative.bag[[j]]<-vector()
success.standard.decision.positive.bag.mod[[j]]<-vector()
success.standard.decision.negative.bag.mod[[j]]<-vector()
success.standard.decision.all.mod[[j]]<-vector()

success.decision.c.opt.positive.bag[[j]]<-vector()
error.decision.c.opt.positive.bag[[j]]<-vector()
error.decision.c.opt.negative.bag[[j]]<-vector()
success.decision.c.opt.negative.bag[[j]]<-vector()
success.decision.c.opt.all[[j]]<-vector()

Delta.success.positive.bag[[j]]<-vector()
Delta.success.negative.bag[[j]]<-vector()

    for (k in 1:dim(vector.p.q)[1])
{

     
set.seed(k)
set.seed.vector.2<-sample(1:(100*num.positive.bags),size=(M[j]*num.positive.bags),replace=FALSE)
set.seed(k)
set.seed.vector.3<-sample(1:(100*num.negative.bags),size=(M[j]*num.negative.bags),replace=FALSE) 
  
positive.bag.noisy[[j]]<-list()
for (i in 1:num.positive.bags)
{positive.bag.noisy[[j]][[i]]<-positive.bag[[j]]
for (r in 1:M[j])
{if (positive.bag[[j]][r]==1)
{set.seed(set.seed.vector.2[r*i])
  positive.bag.noisy[[j]][[i]][[r]]<-sample(c(0,1),1,prob=c(vector.p.q[k,1],1-vector.p.q[k,1]))} else {set.seed(set.seed.vector.2[r*i])
    positive.bag.noisy[[j]][[i]][[r]]<-sample(c(1,0),1,prob=c(vector.p.q[k,2],1-vector.p.q[k,2]))}
}
}




negative.bag.noisy[[j]]<-list()
for (i in 1:num.negative.bags)
{negative.bag.noisy[[j]][[i]]<-negative.bag[[j]]
for (r in 1:M[j])
{set.seed(set.seed.vector.3[r*i])
    negative.bag.noisy[[j]][[i]][[r]]<-sample(c(1,0),1,prob=c(vector.p.q[k,2],1-vector.p.q[k,2]))}
}


########
n.simulated.positive.bag.noisy[[j]]<-list()
for (i in 1:num.positive.bags)
{set.seed(set.seed.vector.2[M[j]*i])
  n.simulated.positive.bag.noisy[[j]][[i]]<-sample(positive.bag.noisy[[j]][[i]],n[j],replace=FALSE)
}



n.simulated.negative.bag.noisy[[j]]<-list()
for (i in 1:num.negative.bags)
{set.seed(set.seed.vector.3[M[j]*i])
  n.simulated.negative.bag.noisy[[j]][[i]]<-sample(negative.bag.noisy[[j]][[i]],n[j],replace=FALSE)
}

#########

pi.simulated.positive.bag[[j]]<-vector()
for (i in 1:num.positive.bags)
{pi.simulated.positive.bag[[j]][i]<-mean(n.simulated.positive.bag.noisy[[j]][[i]])}

pi.simulated.negative.bag[[j]]<-vector()
for (i in 1:num.negative.bags)
{pi.simulated.negative.bag[[j]][i]<-mean(n.simulated.negative.bag.noisy[[j]][[i]])}


#########


m.plus.simulated.positive.bag[[j]]<-vector()
m.minus.simulated.positive.bag[[j]]<-vector()
for (i in 1:num.positive.bags)
{m.plus.simulated.positive.bag[[j]][i]<-as.numeric(m.plus.minus(M[j],vector.p.q[k,1],vector.p.q[k,2],pi.simulated.positive.bag[[j]][i])[[2]][1])
m.minus.simulated.positive.bag[[j]][i]<-as.numeric(m.plus.minus(M[j],vector.p.q[k,1],vector.p.q[k,2],pi.simulated.positive.bag[[j]][i])[[2]][2])
}


m.plus.simulated.negative.bag[[j]]<-vector()
m.minus.simulated.negative.bag[[j]]<-vector()
for (i in 1:num.negative.bags)
{m.plus.simulated.negative.bag[[j]][i]<-as.numeric(m.plus.minus(M[j],vector.p.q[k,1],vector.p.q[k,2],pi.simulated.negative.bag[[j]][i])[[2]][1])
m.minus.simulated.negative.bag[[j]][i]<-as.numeric(m.plus.minus(M[j],vector.p.q[k,1],vector.p.q[k,2],pi.simulated.negative.bag[[j]][i])[[2]][2])
}


#######

result.opt.c.positive.bag[[j]]<-list()
for (i in 1:num.positive.bags)
{result.opt.c.positive.bag[[j]][[i]]<-optimal.c(M[j],vector.p.q[k,1],vector.p.q[k,2],
                                              alpha,beta,m.plus.simulated.positive.bag[[j]][i],m.minus.simulated.positive.bag[[j]][i])
}


result.opt.c.negative.bag[[j]]<-list()
for (i in 1:num.negative.bags)
{result.opt.c.negative.bag[[j]][[i]]<-optimal.c(M[j],vector.p.q[k,1],vector.p.q[k,2],
                                                alpha,beta,m.plus.simulated.negative.bag[[j]][i],m.minus.simulated.negative.bag[[j]][i])
}


#########
### Decision about positive.bag.noisy and negative.bag.noisy with standard procedure (c=1)

standard.decision.c.1.positive.bag[[j]]<-vector()    # 1 = +, 0 = -
for (i in 1:num.positive.bags)
{ if (sum(positive.bag.noisy[[j]][[i]])==0)
{standard.decision.c.1.positive.bag[[j]][i]<-0} else {standard.decision.c.1.positive.bag[[j]][i]<-1}
}


standard.decision.c.1.negative.bag[[j]]<-vector()    # 1 = +, 0 = -
for (i in 1:num.negative.bags)
{ if (sum(negative.bag.noisy[[j]][[i]])==0)
{standard.decision.c.1.negative.bag[[j]][i]<-0} else {standard.decision.c.1.negative.bag[[j]][i]<-1}
}




success.standard.decision.positive.bag[[j]][k]<-sum(standard.decision.c.1.positive.bag[[j]],na.rm=TRUE)
error.standard.decision.positive.bag[[j]][k]<-length(is.na(standard.decision.c.1.positive.bag[[j]])==FALSE)-
                                         success.standard.decision.positive.bag[[j]][k]


error.standard.decision.negative.bag[[j]][k]<-sum(standard.decision.c.1.negative.bag[[j]],na.rm=TRUE)
success.standard.decision.negative.bag[[j]][k]<-length(is.na(standard.decision.c.1.negative.bag[[j]])==FALSE)-
  error.standard.decision.negative.bag[[j]][k]


#########
### Decision aboutpositive.bag.noisy and negative.bag.noisy with c-rule, c given by result.opt.c


decision.c.opt.positive.bag[[j]]<-vector()    # 1 = +, 0 = -
for (i in 1:num.positive.bags)
{ if (is.na(as.numeric(result.opt.c.positive.bag[[j]][[i]][5,2]))==FALSE)
{ if(sum(positive.bag.noisy[[j]][[i]]) < as.numeric(result.opt.c.positive.bag[[j]][[i]][5,2]))
{decision.c.opt.positive.bag[[j]][i]<-0} else {decision.c.opt.positive.bag[[j]][i]<-1}
} else {decision.c.opt.positive.bag[[j]][i]<-NA}
}



decision.c.opt.negative.bag[[j]]<-vector()    # 1 = +, 0 = -
for (i in 1:num.negative.bags)
{ if (is.na(as.numeric(result.opt.c.negative.bag[[j]][[i]][5,2]))==FALSE)
{ if(sum(negative.bag.noisy[[j]][[i]]) < as.numeric(result.opt.c.negative.bag[[j]][[i]][5,2]))
{decision.c.opt.negative.bag[[j]][i]<-0} else {decision.c.opt.negative.bag[[j]][i]<-1}
} else {decision.c.opt.negative.bag[[j]][i]<-NA}
}



success.decision.c.opt.positive.bag[[j]][k]<-sum(decision.c.opt.positive.bag[[j]],na.rm=TRUE)
error.decision.c.opt.positive.bag[[j]][k]<-length(is.na(decision.c.opt.positive.bag[[j]])==FALSE)-
  success.decision.c.opt.positive.bag[[j]][k]


error.decision.c.opt.negative.bag[[j]][k]<-sum(decision.c.opt.negative.bag[[j]],na.rm=TRUE)
success.decision.c.opt.negative.bag[[j]][k]<-length(is.na(decision.c.opt.negative.bag[[j]])==FALSE)-
  error.decision.c.opt.negative.bag[[j]][k]

success.decision.c.opt.all[[j]]<-success.decision.c.opt.positive.bag[[j]]+
                                          success.decision.c.opt.negative.bag[[j]]

summary(success.decision.c.opt.all[[j]])
sd(success.decision.c.opt.all[[j]])

#

no.na.positive.bag<-which(is.na(decision.c.opt.positive.bag[[j]])==FALSE)
success.standard.decision.positive.bag.mod[[j]][k]<-sum(standard.decision.c.1.positive.bag[[j]][no.na.positive.bag])

no.na.negative.bag<-which(is.na(decision.c.opt.negative.bag[[j]])==FALSE)
success.standard.decision.negative.bag.mod[[j]][k]<-length(is.na(decision.c.opt.negative.bag[[j]])==FALSE)-
                     sum(standard.decision.c.1.negative.bag[[j]][no.na.negative.bag])


success.standard.decision.all.mod[[j]]<-success.standard.decision.positive.bag.mod[[j]]+
                                             success.standard.decision.negative.bag.mod[[j]]

summary(success.standard.decision.all.mod[[j]])
sd(success.standard.decision.all.mod[[j]])

#########  COMPARISON STANDARD (c=1) vs c-rule
#success.standard.decision.positive.bag.mod vs, success.decision.c.opt.positive.bag
#success.standard.decision.negative.bag.mod vs, success.decision.c.opt.negative.bag

Delta.success.positive.bag[[j]][k]<-success.decision.c.opt.positive.bag[[j]][k]-
  success.standard.decision.positive.bag.mod[[j]][k]

Delta.success.negative.bag[[j]][k]<-success.decision.c.opt.negative.bag[[j]][k]-
  success.standard.decision.negative.bag.mod[[j]][k]

#Delta.success    # + indica millor c-rule
                  # - indica millor standard rule
                  # 0 indica igual
print(k)
}   # END FOR k

print(j)
}   # END FOR j

#############

    


length(Delta.success.positive.bag[[j]])
length(Delta.success.negative.bag[[j]])

length(which(Delta.success.positive.bag[[j]]<0))
length(which(Delta.success.positive.bag[[j]]==0))
length(which(Delta.success.positive.bag[[j]]>0))

length(which(Delta.success.negative.bag[[j]]<0))
length(which(Delta.success.negative.bag[[j]]==0))
length(which(Delta.success.negative.bag[[j]]>0))


vector.p.q[which(Delta.success.positive.bag[[j]]<0),]
summary(vector.p.q[which(Delta.success.positive.bag[[j]]<0),])
sd(vector.p.q[which(Delta.success.positive.bag[[j]]<0),][[1]])
sd(vector.p.q[which(Delta.success.positive.bag[[j]]<0),][[2]])


vector.p.q[which(Delta.success.negative.bag[[j]]<0),]
summary(vector.p.q[which(Delta.success.negative.bag[[j]]<0),])
sd(vector.p.q[which(Delta.success.negative.bag[[j]]<0),][[1]])
sd(vector.p.q[which(Delta.success.negative.bag[[j]]<0),][[2]])

prop.best.c.rule.positive.bag<-vector()
prop.best.c.rule.positive.bag[j]<-length(which(Delta.success.positive.bag[[j]]>0))/(length(which(Delta.success.positive.bag[[j]]!=0)))
prop.best.c.rule.negative.bag<-vector()
prop.best.c.rule.negative.bag[j]<-length(which(Delta.success.negative.bag[[j]]>0))/(length(which(Delta.success.negative.bag[[j]]!=0)))


length(success.standard.decision.positive.bag.mod[[j]])
length(success.standard.decision.negative.bag.mod[[j]])
length(success.decision.c.opt.positive.bag[[j]])
length(success.decision.c.opt.negative.bag[[j]])

mean(success.standard.decision.positive.bag.mod[[j]])
mean(success.standard.decision.negative.bag.mod[[j]])
mean(success.decision.c.opt.positive.bag[[j]])
mean(success.decision.c.opt.negative.bag[[j]])

summary(success.standard.decision.positive.bag.mod[[j]])
summary(success.standard.decision.negative.bag.mod[[j]])
summary(success.decision.c.opt.positive.bag[[j]])
summary(success.decision.c.opt.negative.bag[[j]])
summary(Delta.success.positive.bag[[j]])
summary(Delta.success.negative.bag[[j]])

sd(success.standard.decision.positive.bag.mod[[j]])
sd(success.standard.decision.negative.bag.mod[[j]])
sd(success.decision.c.opt.positive.bag[[j]])
sd(success.decision.c.opt.negative.bag[[j]])
sd(Delta.success.positive.bag[[j]])
sd(Delta.success.negative.bag[[j]])


cor.test(x = success.standard.decision.positive.bag.mod[[j]], y = success.decision.c.opt.positive.bag[[j]], 
         method = c("pearson"), 
         conf.level = 0.95)

cor.test(x = success.standard.decision.negative.bag.mod[[j]], y = success.decision.c.opt.negative.bag[[j]], 
         method = c("pearson"), 
         conf.level = 0.95)

####

result.positive.bag<-list()
result.positive.bag[[j]]<-as.data.frame(cbind(success.standard.decision.positive.bag.mod[[j]],
                                         success.decision.c.opt.positive.bag[[j]]))
colnames(result.positive.bag[[j]])<-c("standard", "c-rule")

par(pty = "s")
boxplot(result.positive.bag[[j]],
        col = c("#003C67FF", "#EFC000FF"),
        main = "Comparison of rules. Scenario 2. Positive bags. M = 50",
        xlab = "rule", ylab = "Number of successes")




result.negative.bag<-list()
result.negative.bag[[j]]<-as.data.frame(cbind(success.standard.decision.negative.bag.mod[[j]],
                                              success.decision.c.opt.negative.bag[[j]]))
colnames(result.negative.bag[[j]])<-c("standard", "c-rule")

par(pty = "s")
boxplot(result.negative.bag[[j]],
        col = c("#003C67FF", "#EFC000FF"),
        main = "Comparison of rules. Scenario 2. Negative bags. M = 50",
        xlab = "rule", ylab = "Number of successes")




# test of equality of variances
bartlett.test(result.positive.bag[[j]])

t.test(x=result.positive.bag[[j]][[1]], y=result.positive.bag[[j]][[2]],
       alternative = "greater",
       mu = 0, 
       paired = TRUE,   
       var.equal = FALSE,
       conf.level = 0.95)



# test of equality of variances
bartlett.test(result.negative.bag[[j]])

t.test(x=result.negative.bag[[j]][[1]], y=result.negative.bag[[j]][[2]],
       alternative = "less",
       mu = 0, 
       paired = TRUE,   
       var.equal = FALSE,
       conf.level = 0.95)


### Join positive and negative bags
Delta.success.join<-list()
Delta.success.join[[j]]<-vector()
Delta.success.join[[j]]<-Delta.success.positive.bag[[j]]+Delta.success.negative.bag[[j]]

length(Delta.success.join[[j]])

summary(Delta.success.join[[j]])
sd(Delta.success.join[[j]])

length(which(Delta.success.join[[j]]<0))
length(which(Delta.success.join[[j]]==0))
length(which(Delta.success.join[[j]]>0))

vector.p.q[which(Delta.success.join[[j]]<0),]
summary(vector.p.q[which(Delta.success.join[[j]]<0),])
sd(vector.p.q[which(Delta.success.join[[j]]<0),][[1]])
sd(vector.p.q[which(Delta.success.join[[j]]<0),][[2]])


par(pty = "s")
boxplot(vector.p.q[which(Delta.success.join[[j]]<0),],
        col = c("#003C67FF", "#EFC000FF"),
        main = "Scenario 2. M = 50. Distribution of p and q for Delta<0")
        #xlab = " ", ylab = " ")


par(pty = "s")
boxplot(vector.p.q,
        col = c("#003C67FF", "#EFC000FF"),
        main = "Scenario 2. Distribution of p and q")
#xlab = " ", ylab = " ")

prop.best.c.rule.join<-vector()
prop.best.c.rule.join[j]<-length(which(Delta.success.join[[j]]>0))/(length(which(Delta.success.join[[j]]!=0)))



###### HEATMAP OF Delta.success.join with respect to p and q

Delta.success.join.df<-list()
Delta.success.join.df[[j]]<-as.data.frame(cbind(vector.p.q,Delta.success.join[[j]]))
colnames(Delta.success.join.df[[j]])<-c("p","q","Increment")
Delta.success.join.df[[j]]

min<-vector()
Max<-vector()
med<-vector()

min[j]<-min(Delta.success.join.df[[j]][[3]])
Max[j]<-max(Delta.success.join.df[[j]][[3]])
med[j]<-median(Delta.success.join.df[[j]][[3]])


heat.map.Delta.success.scenario.2<-list()
heat.map.Delta.success.scenario.2[[j]]<-ggplot(data = Delta.success.join.df[[j]], aes(x=p, y=q, fill=Increment)) + 
  geom_tile()+
  ggtitle("Scenario 2. M = 50. alpha=0.05, beta=0.10") +
  scale_fill_gradient2(low = "#FF0033", high = "#6666FF", mid = "white", # #FF0033
                       midpoint = 0, limit = c(min[j],Max[j]), space = "Lab", 
                       name="Increment") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))


print(heat.map.Delta.success.scenario.2[[j]])


###### HEATMAP OF Delta.success.positive.bag, with respect to p and q

Delta.success.positive.bag.df<-list()
Delta.success.positive.bag.df[[j]]<-as.data.frame(cbind(vector.p.q,Delta.success.positive.bag[[j]]))
colnames(Delta.success.positive.bag.df[[j]])<-c("p","q","Increment")
Delta.success.positive.bag.df[[j]]

min<-vector()
Max<-vector()
med<-vector()

min[j]<-min(Delta.success.positive.bag.df[[j]][[3]])
Max[j]<-max(Delta.success.positive.bag.df[[j]][[3]])
med[j]<-median(Delta.success.positive.bag.df[[j]][[3]])


heat.map.Delta.success.scenario.2.positive.bag<-list()
heat.map.Delta.success.scenario.2.positive.bag[[j]]<-ggplot(data = Delta.success.positive.bag.df[[j]], 
                                                            aes(x=p, y=q, fill=Increment)) + 
  geom_tile()+
  ggtitle("Scenario 2. M = 50. Positive.bags. alpha=0.05, beta=0.10") +
  scale_fill_gradient2(low = "#FF0033", high = "#6666FF", mid = "white", # #FF0033
                       midpoint = 0, limit = c(min[j],Max[j]), space = "Lab", 
                       name="Increment") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))


print(heat.map.Delta.success.scenario.2.positive.bag[[j]])


###### HEATMAP OF Delta.success.negative.bag, with respect to p and q

Delta.success.negative.bag.df<-list()
Delta.success.negative.bag.df[[j]]<-as.data.frame(cbind(vector.p.q,Delta.success.negative.bag[[j]]))
colnames(Delta.success.negative.bag.df[[j]])<-c("p","q","Increment")
Delta.success.negative.bag.df[[j]]

min<-vector()
Max<-vector()
med<-vector()

min[j]<-min(Delta.success.negative.bag.df[[j]][[3]])
Max[j]<-max(Delta.success.negative.bag.df[[j]][[3]])
med[j]<-median(Delta.success.negative.bag.df[[j]][[3]])


heat.map.Delta.success.scenario.2.negative.bag<-list()
heat.map.Delta.success.scenario.2.negative.bag[[j]]<-ggplot(data = Delta.success.negative.bag.df[[j]], 
                                                            aes(x=p, y=q, fill=Increment)) + 
  geom_tile()+
  ggtitle("Scenario 2. M = 50. Negative.bags. alpha=0.05, beta=0.10") +
  scale_fill_gradient2(low = "#FF0033", high = "#6666FF", mid = "white", # #FF0033
                       midpoint = 0, limit = c(min[j],Max[j]), space = "Lab", 
                       name="Increment") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))


print(heat.map.Delta.success.scenario.2.negative.bag[[j]])


