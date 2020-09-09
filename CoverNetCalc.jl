using Plots
using LightGraphs, SimpleWeightedGraphs, GraphPlot
include("BasisFunctionTypes.jl")
include("Place.jl")

#need to make these variables in scope of the current WS
nx=Int64(length(z))
mss=Float64(Inf)
λ=Array{Float64,1}[]
δ=Array{Float64,1}[]
#valid penalty criteria
Schwarz = :(nx*log(mss)+nk*log(nx))
Akaike = :(nx*log(mss)+2*nk)
Rissanen = :(description(mss,λ,δ,nx)) #Rissanen desciption length
Model30 = :(-nk*(nk<=30))
#nx is # of observation (length of data)
#nk is # of parameters (basis functions in model)
#mss is the mean-sum-square model prediction error
#λ are the model parameters and δ their precisions

options=Dict("stopstep"=>10,
    "testdatum"=> 8000,
    "functions"=> (gaussian,tophat),
    "embedding" => Place.vembed([0,1,6,12,18,24,36]),
 #   "embedding" => ([0, 1], [0, 1, 2, 3]),
    "penalty"=> Rissanen,
    "nbasis" => 2000,
    "nneighbours"=> 1
    )
bs=0.005:0.005:1
msr=Array{Any,2}(undef,length(bs),6)

#Threads.@threads
for bi in 1:length(bs)
    b=bs[bi]

    z=rosslerpoints(5000,0.05,b)
    #plot(z[1,:],z[2,:],z[3,:])
    zn=addnoise(z,0.01)
    #plot(z[1,:])
    #plot!(zn[1,:],linetype=:dots,markersize=0.5)
    yn=zn[1,:]
    mymodel, X, zout, mdlv = Place.buildmodel(yn,options)
    #plot(mdlv)

    yp, yt = Place.freerun(mymodel,yn,5000)
    #plot(yt)
    #plot!(yp,xlimit=(0,5000))

    X,z,Xp,zp=Place.embedding(yn,mymodel.embedv)
    ϕ,offset = Place.placebo(X,mymodel.rbf,false,false)

    indx=argmax(ϕ,dims=2)
    nodes=[i[2] for i in indx]

    g=SimpleWeightedDiGraph(length(mymodel.rbf))

    nd=length(nodes)
    for i in 2:nd
        add_edge!(g,nodes[i-1],nodes[i],weights(g)[nodes[i-1],nodes[i]]+1)
    end

    A=Array(g.weights)
    am=A+A'
    pg=Graph(am)
    pgis=connected_components(pg)
    pgi=pgis[argmax(length.(pgis))]

    #gplot(pg[pgi],nodelabel=pgi,edgelinewidth=log.(am[am.>0]).+1) #not sure that the weights are in the right order...
    msr[bi,1]=diameter(pg[pgi])
    msr[bi,2]=maxsimplecycles(g)
    msr[bi,3]=global_clustering_coefficient(g)
    gww=g.weights[g.weights.>0]
    gww=gww./sum(gww)
    msr[bi,4]=sum(gww.*log.(gww))
    msr[bi,5]=length(pgi)
    msr[bi,6]=size(g)[1]

end
