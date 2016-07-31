# Run Experiments, display results (and possibly save data) on a Bandit Problem to be specified

using HDF5

# DO YOU WANT TO SAVE RESULTS?
typeExp = "Save"
# typeExp = "NoSave"

# TYPE OF DISTRIBUTION 
@everywhere typeDistribution="Bernoulli"
@everywhere include("BAIalgos.jl")

# CHANGE NAME (save mode)
fname="results/Experiment4arms"

# BANDIT PROBLEM
@everywhere mu=vec([0.3 0.25 0.2 0.1])
@everywhere best=find(mu.==maximum(mu))[1]
K=length(mu)

# RISK LEVEL
delta=0.1

# NUMBER OF SIMULATIONS
N=100


# OPTIMAL SOLUTION
@everywhere v,optWeights=OptimalWeights(mu)
@everywhere gammaOpt=optWeights[best]
print("mu=$(mu)\n")
print("Theoretical number of samples: $((1/v)*log(1/delta))\n")
print("Optimal weights: $(optWeights)\n\n")

# POLICIES 

@everywhere ChernoffPTSHalf(x,y,z)=ChernoffPTS(x,y,z,0.5)
@everywhere ChernoffPTSOpt(x,y,z)=ChernoffPTS(x,y,z,gammaOpt)

policies=[TrackAndStop,ChernoffBC2,ChernoffPTSHalf,ChernoffPTSOpt,KLLUCB,UGapE]
names=["TrackAndStop","ChernoffBC","ChernoffPTS","ChernoffPTSOpt","KLLUCB","UGapE"]


# EXPLORATION RATES 
@everywhere explo(t,n,delta)=log((log(t)+1)/delta)

lP=length(policies)
rates=[explo for i in 1:lP]



# RUN EXPERIMENTS

function MCexp(mu,delta,N)
	for imeth=1:lP
            policy=policies[imeth]
            beta=rates[imeth]
	    startTime=time()
	    a = Array(RemoteRef, N)
	    for j in 1:N
		a[j] =  @spawn policy(mu,delta,beta)
	    end
	    res = collect([fetch(a[j]) for j in 1:N])
            proportion = zeros(N,K)
            for j in 1:N
                n=res[j][2]
                proportion[j,:]=n/sum(n)
            end
            NonTerminated = sum(collect([res[j][1]==0 for j in 1:N]))
            FracNT=(NonTerminated/N)
            FracReco=zeros(K)
            for k in 1:K
                FracReco[k]=sum([(res[j][1]==k)?1:0 for j in 1:N])/(N*(1-FracNT))
            end
            print("Results for $(policy), average on $(N) runs\n")  
	    print("proportion of runs that did not terminate: $(FracNT)\n") 	 	    
	    print("average number of draws: $(sum([sum(res[j][2]) for j in 1:N])/(N-NonTerminated))\n")
            print("average proportion of draws: \n $(mean(proportion,1))\n")
	    print("proportion of errors: $(1-sum([res[j][1]==best for j in 1:N])/(N-NonTerminated))\n")
            print("proportion of recommendation made when termination: $(FracReco)\n")
            print("elapsed time: $(time()-startTime)\n\n")
	    print("")
	end
end


function SaveData(mu,delta,N)
        K=length(mu)
        for imeth=1:lP
            Draws=zeros(N,K)
            policy=policies[imeth]
            namePol=names[imeth]
            startTime=time()
	    a = Array(RemoteRef, N)
            rate=rates[imeth]
	    for j in 1:N
		a[j] =  @spawn policy(mu,delta,rate)
	    end
	    res = [fetch(a[j]) for j in 1:N]
            proportion=zeros(N,K)
            for k in 1:N
               r=res[k][2]
               Draws[k,:]=res[k][2]
               proportion=r/sum(r)
            end
            Reco=[res[j][1] for j in 1:N]
            Error=collect([(r==best)?0:1 for r in Reco])
            FracNT=sum([r==0 for r in Reco])/N
            FracReco=zeros(K)
            for k in 1:K
                FracReco[k]=sum([(r==k)?1:0 for r in Reco])/(N*(1-FracNT))
            end
            print("Results for $(policy), average on $(N) runs\n") 
	    print("proportion of runs that did not terminate: $(FracNT)\n")  	    
	    print("average number of draws: $(sum(Draws)/(N*(1-FracNT)))\n")
            print("average proportions of draws: $(mean(proportion,1))\n")
	    print("proportion of errors: $(sum(Error)/(float(N*(1-FracNT))))\n")
            print("proportion of recommendation made when termination: $(FracReco)\n")
            print("elapsed time: $(time()-startTime)\n\n")
            name="$(fname)_$(namePol)_delta_$(delta)_N_$(N).h5"
            h5write(name,"mu",mu)
            h5write(name,"delta",delta)
            h5write(name,"FracNT",collect(FracNT))
            h5write(name,"FracReco",FracReco)
            h5write(name,"Draws",Draws)
            h5write(name,"Error",Error)
	end
end


if (typeExp=="Save")
   SaveData(mu,delta,N)
else
   MCexp(mu,delta,N)
end

