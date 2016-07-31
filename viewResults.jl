# Visualize results that are saved in the results folder (names should match)

using PyPlot
using HDF5

# NAME AND POLICIES NAMES
fname="results/Experiment4arms"
names=["TrackAndStop","ChernoffBC","ChernoffPTS","ChernoffPTSOpt","KLLUCB","UGapE"]

# PARAMETERS
delta=0.1
N=100

clf()

# personalize bins
xdim=3
ydim=2

NBins=30
xmax=10000


xtxt=0.6*xmax
Bins=round(Int,linspace(1,xmax,NBins))

mu=h5read("$(fname)_$(names[1])_delta_$(delta)_N_$(N).h5","mu")
K=length(mu)

clf()
title("mu = $(mu)")

for j in 1:length(names)
    name="$(fname)_$(names[j])_delta_$(delta)_N_$(N).h5"
    FracNT=h5read(name,"FracNT")
    Draws=h5read(name,"Draws")
    Error=h5read(name,"Error")
    subplot(xdim,ydim,j)
    NbDraws=sum(Draws,2)'
    proportion=zeros(N,K)
    for k in 1:N
        proportion[k,:]=Draws[k,:]/sum(Draws[k,:])
    end
    prop=mean(proportion,1)
    MeanDraws=mean(NbDraws)
    StdDraws=std(NbDraws)
    histo=plt[:hist](vec(NbDraws),Bins)
    Mhisto=maximum(histo[1])
    PyPlot.axis([0,xmax,0,Mhisto])
    ytxt1=0.75*Mhisto
    ytxt2=0.6*Mhisto
    ytxt3=0.5*Mhisto
    ytxt4=0.4*Mhisto
    EmpError=round(Int,10000*mean(Error))/10000
    FracReco=round(Int,1000*prop)/1000
    axvline(MeanDraws,color="black",linewidth=2.5)
    PyPlot.text(xtxt,ytxt1,"mean = $(round(Int,MeanDraws)) (std=$(round(Int,StdDraws)))")
    PyPlot.text(xtxt,ytxt2,"delta = $(delta)")
    PyPlot.text(xtxt,ytxt3,"emp. error = $(EmpError)")
    PyPlot.text(xtxt,ytxt4,"emp. proportions = $(FracReco)")
    if (j==1)
       title("mu=$(mu), $(names[j])")	
    else 	    	    
       title("$(names[j])")	
    end
    print("Results for $(names[j]), average on $(N) runs\n")  
    print("proportion of runs that did not terminate: $(FracNT)\n") 	 	    
    print("average number of draws: $(MeanDraws)\n")
    print("average proportion of draws: \n $(prop)\n")
    print("proportion of errors: $(EmpError)\n")
    print("proportion of recommendation made when termination: $(FracReco)\n\n")
 
end
