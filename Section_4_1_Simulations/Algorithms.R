
################################################################################
#######.  ALGORITHMS for the paper:
#######. 
#######.  A Simple Approach to Multiple Instance Learning:
#######.  Controlling the False Positive Rate
#######.     
#######.  by Rosario Delgado
#######.  May, 2024
################################################################################


###################
###################
####  ALGORITHM 1 #
###################
###################
##
## INPUTS:
## M = bag size
## p = false negative rate
## q = false positive rate
##
## alpha = threshold for P(Type I error)
## beta = threshold for P(Type II error)
##
## m.plus = proportion of genuinely positive instances in the bag
## m.minus = proportion of genuinely negative instances in the bag
## (m.minus = M-m.plus)
## Can be estimated by ALGORITHM 2
##
##
## OUTPUTS:
## c1 = minimum value of c such that P(type I error) <= alpha
## Prob Type I error at c1 (it must be <= alpha)
## Prob Type II error at c1 (if it is <= beta, optimal c = c1)
##
## optimal c = estimated optimal value of c for the "c-rule" 
## Prob Type I error at optimal c (it must be <= alpha)
## Prob Type II error at optimal c (it must be <= beta)
##
## Standard prob error type I (corresponding to c=1). To compare.
## Standard prob error type II (corresponding to c=1). To compare.


varphi<-function(c, m.plus,m.minus,p,q)  # auxiliar function varphi(c)=P_c(type II error)
{M<-m.plus+m.minus
if (round(c)!=c | c<1 | c > M)
  {print("Error in input in varphi(c)")
  suma<-NULL
  stop} else {suma<-0
for (i in (max(c-1-m.plus,0)):min(c-1,m.minus))
{ suma.p<-pbinom(i-(c-1-m.plus)-1, size = m.plus, prob = p, lower.tail = FALSE)
  suma<-suma+choose(m.minus,i)*q^i*(1-q)^{m.minus-i}*suma.p}
  }
return(suma)
}

################


optimal.c<-function(M,p,q,alpha,beta,m.plus,m.minus)   # find the optimal c: ALGORITHM 1
                                                       # uses function "varphi" introduced before
{ standard.prob.error.I<-pbinom(0, size = M, prob = q, lower.tail = FALSE)   # for c=1
  standard.prob.error.II<-varphi(c=1,m.plus,m.minus,p,q) # for c=1

  if (round(M)!=M | M < 1 | round(m.plus)!=m.plus | m.plus<0 | round(m.minus) != m.minus 
      | m.minus<0 | m.plus+m.minus!= M | 
      p < 0 | p >= 0.5 | q < 0 | q >= 0.5 | 
      alpha < 0 | alpha > 1| beta < 0 | beta > 1)
{print("Error in input")
  opt.c<-NULL
  prob.error.I.opt.c<-NULL 
  prob.error.II<-NULL
  c1<-NULL
  prob.error.I.c.1<-NULL
  prob.error.II.c.1<-NULL
  result<-NULL
  #stop
  } else {
    if (q^M > alpha)
    {print("(h1) fails. No feasible solution for P_c(type I error)<=alpha")
      opt.c<-NULL
      prob.error.I.opt.c<-NULL 
      prob.error.II<-NULL
      c1<-NULL
      prob.error.I.c.1<-NULL
      prob.error.II.c.1<-NULL
      result<-NULL
      # stop
      } else { # First, we find c1, which exists if q^M <= alpha
                c1<-qbinom(alpha,M,q,lower.tail=FALSE)+1
                prob.error.I.c.1<-1-pbinom(c1-1,M,q,lower.tail=TRUE) 
                prob.error.II.c.1<-varphi(c=c1,m.plus,m.minus,p,q)
      
        if (m.minus==0 & prob.error.II.c.1>beta)
        {print("(h2.0) fails. No feasible solution for P_c(type II error)<=beta")
          opt.c<-NULL
          prob.error.I.opt.c<-NULL 
          prob.error.II<-NULL
          result = data.frame(cbind(c("c1","Prob Type I error at c1","Prob Type II error at c1"," ", 
                                      "Standard c=1", 
                                      "Standard prob error type I", "Standard prob error type II"),
                                    c(c1,prob.error.I.c.1,prob.error.II.c.1, " "," ", 
                                      standard.prob.error.I, standard.prob.error.II)))
          # stop
          } else {
        
        if (m.minus > 0 & q^{m.minus-1}*(m.minus*(1-q)+q*(1-(1-p)^{m.plus})) > beta)
        {print("(h2.1) fails. No feasible solution for P_c(type II error)<=beta")
          opt.c<-NULL
          prob.error.I.opt.c<-NULL 
          prob.error.II<-NULL
          result = data.frame(cbind(c("c1","Prob Type I error at c1","Prob Type II error at c1"," ",
                                      "Standard c=1", 
                                      "Standard prob error type I", "Standard prob error type II"),
                                    c(c1,prob.error.I.c.1,prob.error.II.c.1, " "," ", 
                                      standard.prob.error.I, standard.prob.error.II)))
          # stop
          } else { 
        # if (m.minus==0 & p^M<=beta)|(m.minus>0 & q <= 1/(m.minus+1) &  
        #    q^{m.minus-1}*(m.minus*(1-q)+q*(1-(1-p)^{m.plus})) <= beta))
        # Second, we find optimal c >= c1 under (h2.0) an (h2.1)
        
        prob.error.II<-varphi(c=c1,m.plus,m.minus,p,q)
        c<-c1
        while (prob.error.II>beta)
        {prob.error.II<-varphi(c+1,m.plus,m.minus,p,q)
          c<-c+1}
        opt.c<-as.integer(round(c,0))
        
        prob.error.I.opt.c<-pbinom(opt.c-1, size = M, prob = q, lower.tail = FALSE)
        
        result = data.frame(cbind(c("c1","Prob Type I error at c1","Prob Type II error at c1"," ", 
                                    "optimal c","Prob Type I error at optimal c",
                                    "Prob Type II error at optimal c"," ", "Standard c=1",
                                    "Standard prob error type I", "Standard prob error type II"),
                                  c(c1,prob.error.I.c.1,prob.error.II.c.1," ",
                                    opt.c,prob.error.I.opt.c, prob.error.II, " ", " ", 
                                    standard.prob.error.I, standard.prob.error.II)))
            }
        }
      }
  }

 
rownames(result) <- NULL
colnames(result) <- NULL
  
  
return(result)
    }
    
######### END OF FUNCTION optimal.c

###################
###################
####  ALGORITHM 2 #
###################
###################
##
## INPUTS:
## M = bag size
## p = false negative rate
## q = false positive rate
## pi = proportion of positively classified instances in a random sambple of the bag
##
## OUTPUTS:
## m.plus = estimated number of genuinely positive instances in the bag
## m.minus = estimated number of genuinely negative instances in the bag
## m.plus + m.minus = M

m.plus.minus<-function(M,p,q,pi)
{ if (round(M)!=M | M < 1 | p < 0 | p >= 0.5 | q < 0 | q >=0.5 | pi < 0 | pi > 1)
{print("Error in input")
  result<-NULL
  stop} else {
    if (pi < q/(1-p))
    {m.plus=0
    m.minus=M
    stop} else {m.plus<-round(M*(pi*(1-p)-q)/(1-p-q),0)
    m.minus<-M-m.plus}
    #return(c("m.plus=",m.plus,"m.minus=",m.minus))
    result = data.frame(cbind(c("m.plus","m.minus"),c(m.plus,m.minus)))
    rownames(result) <- NULL
    colnames(result) <- NULL
  }
  return(result)
}



