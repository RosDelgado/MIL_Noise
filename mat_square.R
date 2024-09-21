
################################################################################
#######.  A simple algorithm that converts a confusion matrix which is not square
#######.  into square, by adding rows or columns of zero values, as appropriate
################################################################################


mat.square<-function(M=matrix, c=labels)  # input: the original confusion matrix, M
                              # desired vector of labels for rows and for columns, c
  { 
  cc<-length(c)
  N <- matrix(c(rep(0,cc^2)),nrow=cc)
  rownames(N)<-c
  colnames(N)<-c
  
  if (!is.vector(M))
  {
    for (i in 1:cc)
    {for (j in 1:cc)
    {   
      for (k in 1:nrow(M))
      {
        for (m in 1:ncol(M))
        {
          if (rownames(M)[k]==c[i] & colnames(M)[m]==c[j])
          {N[i,j]<-M[k,m]}
        }
      }
    }
    }
  } else
  {
    for (i in 1:cc)
    {for (j in 1:cc)
    {   
      for (m in 1:ncol(M))
      {
        if (rownames(M)==c[i] & colnames(M)[m]==c[j])
        {N[i,j]<-M[k,m]}
      }
      
    }
    }
  }
  
  # output: N, the modified square confusion matrix obtained from M 
  
  return(N)
  
}
