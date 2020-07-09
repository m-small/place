
using DifferentialEquations, Statistics

function lorenz!(du,u,p,t)
	#standard Lorenz equations
    x,y,z = u
    σ,ρ,β = p
    ####################
    du[1] = dx = σ*(y-x)
    du[2] = dy = x*(ρ-z) - y
    du[3] = dz = x*y - β*z
end

function lorenzpoints(np,sr=0.05)
	#np points every sr sampling steps
	param=[10.0, 28.0, 8/3]
	#transient for "long enough"
	u0 = [1.0;0.0;0.0]
	tspan = (0.0,1.0)
	prob = ODEProblem(lorenz!,u0,tspan,param)
	sol=solve(prob)
	u0=sol.u[end]
    #simulate	
	tspan = (0.0, np*sr)
	prob = ODEProblem(lorenz!,u0,tspan,param,saveat = sr)
	sol=solve(prob)
	z=sol[:,:]
	return z
end

function rossler!(du,u,p,t)
	#standard Rossler equations
	x,y,z=u
	a,b,c=p
	#################
	du[1]=-y-z
	du[2]=x+a*y
	du[3]=b+z*(x-c)
end


function rosslerpoints(np,sr=0.05)
	#np points every sr sampling steps
	param=[0.2 0.2 5.7]
	#transient for "long enough"
	u0 = [1.0;0.0;0.0]
	tspan = (0.0,10.0)
	prob = ODEProblem(rossler!,u0,tspan,param)
	sol=solve(prob)
	u0=sol.u[end]
    #simulate	
	tspan = (0.0, np*sr)
	prob = ODEProblem(rossler!,u0,tspan,param,saveat = sr)
	sol=solve(prob)
	z=sol[:,:]
	return z
end

function itmap(mapfn,x0::Float64,it)
	#transient for "long enough"
    for i in 1:1000
        x0=mapfn(x0)
    end
    #simulate
    x=Array{Float64,1}(undef,it)
    for i in 1:it
        x0=mapfn(x0)
        x[i]=x0
    end
    return x
end

function itmap(mapfn,x0::Array{Float64,1},it)
    dx=length(x0)
    for i in 1:1000
        x0=mapfn(x0)
    end
    x=Array{Float64,2}(undef,dx,it)
    for i in 1:it
        x0=mapfn(x0)
        x[:,i]=x0
    end
    return x
end

function logistic(x,λ=4.0)
    return λ*x*(1-x)
end

function henon(x)
    y=Array{Float64,1}(undef,2)
    a=1.4
    b=0.3
    y[1]=1-a*x[1]^2+x[2]
    y[2]=b*x[1]
    return y
end

function ikeda(x,u=0.6)
	y=Array{Float64,1}(undef,2)
	θ=0.4-6/(1+x[1]^2+x[2]^2)
	y[1]=1+u*(x[1]*cos(θ)-x[2]*sin(θ))
	y[2]=u*(x[1]*sin(θ)+x[2]*cos(θ))
	return y
end

function tinkerbell(x,a=0.9,b=-0.6013,c=2.0,d=0.50)
	#tinkerbell map
	# other standard values: a=0.3, b=0.6, c=2.0, d=0.27
	y=Array{Float64,1}(undef,2)
	y[1]=x[1]^2-x[2]^2+a*x[1]+b*x[2]
	y[2]=2*x[1]*x[2]+c*x[1]+d*x[2]
	return y
end


function addnoise(y,σ)
	#add observational noise with standard deviation σ/std(y) to y
	stdy=std(y[:])
	z=y+randn(size(y))*stdy*σ
	return z
end





