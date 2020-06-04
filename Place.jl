module Place

	export buildmodel, placebo, BasisFunc, PlaceModel

	using ToeplitzMatrices, Random, StatsBase, Distributions

	function tophat(x,p)
    	return exp.(-x.^p)
	end
	function dtophat(x,p)
    	return -p* (x.^(p-1)) .* exp.(-x.^p)
	end
	function tophat()
		return [[1.1, 5]]
	end

	function gaussian(x) 
    	return exp.(-x.^2)
	end
	function gaussian()
		return [[]]
	end
	function dgaussian(x) 
    	return -2*x .* exp.(-x.^2)
	end


	function cubic(x)
		return x.^3
	end
	function cubic()
		return [[]]
	end
	function dcubic(x)
		return 3 .* x.^2
	end

	struct BasisFunc
    	funct
    	embed
    	radius
    	centre
    	params
	end

	struct PlaceModel
		rbf
		λ
		params
	end

	function buildmodel(y,options=[])
		#build a time series model of y splitting at point td between build and test

		#sort out all the options
		if "nbasis"∈options
			nbasis=options["nbasis"]
		else
			nbasis=500 #default 500
		end
		if "testdatum"∈options
			td=options["testdatum"]
		else
			td=[]
		end
		if "stopstep"∈options
			stopstep=options["stopstep"]
		else
			stopstep=6 #default 6
		end
		if "embedding"∈options
			v=options["embedding"]
		else
			v=[0, 1, 2, 3]
		end
		if v isa Tuple
			nv=length(v)
		else
			nv=1
		end
		if "functions"∈options
			functype=options["functions"]
		else
			functype=(gaussian, tophat)
		end


		#embed the data
		de=maximum([maximum(vv) for vv in v])+1 #maximum embedding lag required
		x0=y[1:(de+1)] #initial condition
		z=y[(de+1):end] #predicted target
		X=ToeplitzMatrices.Hankel(x0,z) #sparse embedded matrix
		X=X[1:end-1,:] #state matrix
		(de,nX)=size(X)
		if isempty(td)
			td=Int(floor(nX/2))
		end
		if td>nX
			td=Int(floor(nX*0.9))
		end
		Xp=X[:, td:end] 
		zp=z[td:end] #Xp and zp are the test data
		X=X[:, 1:(td-1)]
		z=z[1:(td-1)]#X and z are used to build the model

		#generate some basis functions
		rbfset=getbasis(X,z,functype,v, nbasis)

		#choose basis functions
		rbf,ϕ,λ=topdown(X,z,rbfset)

		#build and output model structure
		model=PlaceModel()
		model.rbf=rbf
		model.λ=λ
		model.params=params
		#done
		return model
	end

	function getbasis(X :: Array{Float64,2}, y :: Array{Float64, 1}, functions, v, nbasis) :: Array{BasisFunc,1}
		#generate a random set of basis functions, distributed over the data X according to pdf (X,y)
		de,nx=size(X)
		rbfs=Array{BasisFunc,1}(undef,nbasis)
		randfunct=rand(functions,nbasis)
		if v isa Tuple
			randv=rand(v,nbasis)
		else
			randv=[]
		end
		randi=sample(1:nx, Weights(abs.(y)),nbasis)
		randomneighbour=sample(1:nx, Weights(abs.(y)),nbasis) 
		randr=X[:,randi]-X[:,randomneighbour]
		randr=sum(X.^2, dims=1).^(0.5)
		randr=randr.*randn(nbasis)
		for i in 1:nbasis
			thisfunct=randfunct[i]
			theseparams=[]
			for (pi,pr) in enumerate(thisfunct()) #choose option parameters, if any
				if ~isempty(pr)
					theseparams = [theseparams; rand(Uniform(pr[1],pr[2]))]
				end
			end
			if ~isempty(theseparams)
				println(theseparams)
			end
			if ~isempty(randv)
				thisv=randv[i]
			else
				thisv=v
			end
			rbf=BasisFunc(thisfunct, thisv, randr[i], X[thisv.+1,randi], theseparams)
			rbfs[i]=rbf
		end
		return rbfs
	end

	function topdown(X,y,rbfs)
		#select an optimal set of basis functions to fit X to y.
		rbfs=rbfs[1:10]
		ϕ=placebo(X,rbfs)
		λ=X\ϕ
		#currently, does nothing.
		return ϕ, λ
 	end

	function norm(X,n:: Int64=2)
		#n-norm over columns of a square array X
		return (sum(X.^n, dims=1)).^(1/n)
	end

	function placebo(X :: Array{Float64,2},rbfs ::Array{BasisFunc,1})
#	function placebo(X :: Array{Float64,2}, rbfs):: Array{Float64,2}
		#evaluate the basis functions rbfs at the points of X
		(de,nx)=size(X)
		nb=length(rbfs)
		ϕ=Array{Float64,2}(undef,nx,nb)

		for (i,rbf) in enumerate(rbfs)

			ϕX = X[rbf.embed.+1, : ] - rbf.centre[rbf.embed.+1]*ones(1,nx)
			ϕX = norm(ϕX) ./ rbf.radius
			if isempty(rbf.params)
				ϕi = rbf.funct(ϕX)		
			else
				ϕi = rbf.funct(ϕX, rbf.params)		
			end
			ϕ[:,i] = ϕi

		end

		return ϕ
	end

end # of module