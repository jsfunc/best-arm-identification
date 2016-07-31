# Bernoulli distributions

function dBernoulli(p,q)
  res=0
  if (p!=q)
     if (p<=0) p = eps() end
     if (p>=1) p = 1-eps() end
     res=(p*log(p/q) + (1-p)*log((1-p)/(1-q))) 
  end
  return(res)
end


function dupBernoulli(p,level)
# KL upper confidence bound:
# return qM>p such that d(p,qM)=level 
lM = p 
uM = min(min(1,p+sqrt(level/2)),1) 
for j = 1:16
   qM = (uM+lM)/2
   if dBernoulli(p,qM) > level
      uM= qM
   else
      lM=qM
   end
end
return(uM)
end


function dlowBernoulli(p,level)
# KL lower confidence bound:
# return lM<p such that d(p,lM)=level
lM = max(min(1,p-sqrt(level/2)),0) 
uM = p 
for j = 1:16
   qM = (uM+lM)/2;
   if dBernoulli(p,qM) > level
      lM= qM;
   else
      uM=qM;
   end
end
return(lM)
end


# Poisson distributions

function dPoisson(p,q)
  if (p==0)
     res=q
  else
     res=q-p + p*log(p/q)
  end
  return(res)
end


function dupPoisson(p,level)
# KL upper confidence bound: generic way
# return qM>p such that d(p,qM)=level 
lM = p 
# finding an upper bound
uM = max(2*p,1)
while (dPoisson(p,uM)<level)
     uM=2*uM
end
for j = 1:16
   qM = (uM+lM)/2
   if dPoisson(p,qM) > level
      uM= qM
   else
      lM=qM
   end
end
return(uM)
end


function dlowPoisson(p,level)
# KL lower confidence bound: generic way
# return lM<p such that d(p,lM)=level
# finding a lower bound
lM=p/2
if p!=0
   while (dPoisson(p,lM)<level)
      lM=lM/2
   end
end
uM = p 
for j = 1:16
   qM = (uM+lM)/2;
   if dPoisson(p,qM) > level
      lM= qM;
   else
      uM=qM;
   end
end
return(lM)
end



# Exponential distribution

function dExpo(p,q)
  res=0
  if (p!=q)
     if (p<=0)|(q<=0)
        res=Inf
     else
        res=p/q - 1 - log(p/q)
     end
  end
  return(res)
end

function dupExpo(p,level)
# KL upper confidence bound: generic way
# return qM>p such that d(p,qM)=level 
lM = p 
# finding an upper bound
uM = max(2*p,1)
while (dExpo(p,uM)<level)
     uM=2*uM
end
for j = 1:16
   qM = (uM+lM)/2
   if dExpo(p,qM) > level
      uM= qM
   else
      lM=qM
   end
end
return(uM)
end


function dlowExpo(p,level)
# KL lower confidence bound: generic way
# return lM<p such that d(p,lM)=level
# finding a lower bound
lM=p/2
if p!=0
   while (dExpo(p,lM)<level)
      lM=lM/2
   end
end
uM = p 
for j = 1:16
   qM = (uM+lM)/2;
   if dExpo(p,qM) > level
      lM= qM;
   else
      uM=qM;
   end
end
return(lM)
end


# Gaussian distribution


function dGaussian(p,q)
  (p-q)^2/(2*sigma^2)
end


function dupGaussian(p,level)
   p+sigma*sqrt(2*level)
end

function dlowGaussian(p,level)
   p-sigma*sqrt(2*level)
end


