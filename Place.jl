module Place

	export buildmodel, placebo, predict, freerun, BasisFunc, PlaceModel

	using ToeplitzMatrices, Random, StatsBase, Distributions, LinearAlgebra

	function tophat(x,p)
	    return exp.(-x.^p)
	end
	function dtophat(x,p)
    	return -p* (x.^(p-1)) .* exp.(-x.^p)
	end
	function tophat()
		return [1.1 5;]
	end

	function gaussian(x) 
    	return exp.(-x.^2)
	end
	function gaussian()
		return []
	end
	function dgaussian(x) 
    	return -2*x .* exp.(-x.^2)
	end

	function cubic(x)
		return x.^3
	end
	function cubic()
		return []
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
		basis
		embedv
	end

	function predict(model,y)
		#use the model to predict y
		#
		#embed
		X,yt =embedding(y,model.embedv)
		#compute ϕ
		ϕ, offset = placebo(X, model.rbf)
		#compute prediction
		yp=ϕ[:,model.basis]*model.λ
		ep=yp-yt
		#
		return yp,yt,ep
	end

	function freerun(model,y,npts ::Int64=0,i ::Int64=1)
		#use the model to predict y
		#
		#embed
		X,yt =embedding(y,model.embedv)
		dX,nx =size(X)
		if npts<1
			npts=length(yt)
		end
		Xi=X[:,i]
		Xp=Array{Float64,2}(undef,dX,npts+1)
		for j in 1:npts
			ϕ, offset = placebo(Xi[:,:], model.rbf) #Xi needs to be 2-D!
			yp=ϕ[:,model.basis]*model.λ
			Xi=[yp; Xi[1:(end-1)]]
			Xp[:,j]=Xi
		end
		yp=Xp[1,:]
		#
		return yp
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
		X,z = embedding(y,v,td)
		
		#generate some basis functions
		rbfset = getbasis(X,z,functype,v, nbasis)

		#choose basis functions
		rbf,λ,basis = topdown(X,z,rbfset)

		#build and output model structure
		model = PlaceModel(rbf,λ,params,basis,v)
		#
		#done
		return model, X, z
	end

	function embedding(y,v,td=0)
		#embed y with strategy adequate for v
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
		if td==0
			Xp=[]
			zp=[]
			td=nX+1
		else
			Xp=X[:, td:end] 
			zp=z[td:end] #Xp and zp are the test data
		end
		X=X[:, 1:(td-1)]
		z=z[1:(td-1)]#X and z are used to build the model
		#
		return X, z, Xp, zp
	end

	function getbasis(X :: Array{Float64,2}, y :: Array{Float64, 1}, functions, v, nbasis) :: Array{BasisFunc,1}
		#generate a random set of basis functions, distributed over the data X according to pdf (X,y)
		de,nx=size(X)
		rbfs=Array{BasisFunc,1}(undef,nbasis)
		randfunct=rand(functions,nbasis)
		randi=sample(1:nx, Weights(abs.(y)),nbasis)
		randomneighbour=sample(1:nx, Weights(abs.(y)),nbasis) 
		if v isa Tuple
			randv=rand(v,nbasis)
			randr=X[randv.+1,randi]-X[randv.+1,randomneighbour]
		else
			randv=[]
			randr=X[v.+1,randi]-X[v.+1,randomneighbour]
		end
		randr=normn(randr)
		randr=abs.(randr.*randn(nbasis))
		for i in 1:nbasis
			#
			thisfunct=randfunct[i]
			theseparams=[]
			tfp=thisfunct()
			if ~isempty(tfp)
				np,dp=size(tfp)
				theseparams=Array{Float64,1}(undef,np)
				for npi in 1:np #choose option parameters, if any
					theseparams[npi] = rand(Uniform(tfp[npi,1],tfp[npi,2]))
				end
			end
			if ~isempty(randv)
				thisv=randv[i]
			else
				thisv=v
			end
			rbf=BasisFunc(thisfunct, thisv, randr[i], X[thisv.+1,randi], theseparams)
			rbfs[i]=rbf
			#
		end
		#
		#
		return rbfs
	end

	function normn(X,n:: Int64=2)
		#n-norm over columns of a square array X
		#
		return (sum(X.^n, dims=1)).^(1/n)
	end

	function topdown(X,y,allrbfs)
		#select an optimal set of basis functions to fit X to y.
		ϕ,offset=placebo(X,allrbfs)
		nb=length(allrbfs)
		modelbasis=[]
		ϕQ=Array{Float64,2}(undef,0,0)
		ϕR=Array{Float64,2}(undef,0,0)
#		δ=Array{Float64,2}(undef,0,0)
		err=y
		nk=0
		while nk<30
			#
			μ = -ϕ'*err #compute sensitivity
			#first try an expand
			~,ind = findmax(abs.(μ))
			append!(modelbasis, ind)
			ϕQ,ϕR = qrappend(ϕQ,ϕR,ϕ[:,ind])
			a1 = ϕR\(ϕQ'*y) #new model weights
			#now delete
			~,ind = findmin(abs.(a1))
			if ind==(nk+1) #last added is worst so keep it
	#			append!(δ,mean(δ))
				nk += 1
			else
				#last added is not worst, so delete new worst
				deleteat!(modelbasis,ind)
				ϕQ,ϕR = qrdelete(ϕQ,ϕR,ind)
			end
			#   
			a1 = ϕR\(ϕQ'*y) #new model weights   
			err = y - ϕ[:,modelbasis]*a1
         	#mss= err'*err/dim_y;
		end
		λ = ϕR\(ϕQ'*y) 									

		rbfs = allrbfs[filter(x -> x>offset, modelbasis).-offset]
		mbi = [filter(x->x<=offset, modelbasis); (1:count(modelbasis.>offset)) .+ offset]
		ϕ, = placebo(X,rbfs)		
		λ = ϕ[:,mbi]\y 	
		#
		return rbfs, λ, mbi #do I need to return the ϕ?
 	end

	function placebo(X :: Array{Float64,2}, rbfs ::Array{BasisFunc,1}, constant::Bool=true, linear::Bool=true)
		#evaluate the basis functions rbfs at the points of X
		#include constant and linear terms in the model candidates
		(de,nx)=size(X)
		nb=length(rbfs)
		offset=Int64(constant)+Int64(linear)*de
		ϕ=Array{Float64,2}(undef,nx,nb+offset)
		if constant
			ϕ[:,1] .= 1
		end
		if linear
			ϕ[:,2:(de+1)]=X'
		end
		for (i,rbf) in enumerate(rbfs)
			#
			ϕX = X[rbf.embed.+1, : ] - rbf.centre[rbf.embed.+1]*ones(1,nx)
			ϕX = normn(ϕX) ./ rbf.radius
			if isempty(rbf.params)
				ϕi = rbf.funct(ϕX)		
			else
				ϕi = rbf.funct(ϕX, rbf.params)		
			end
			ϕ[:,i+offset] = ϕi
			#
		end
		#
		#
		return ϕ, offset
	end

	function qrappend(Q,R,x)
		# [Q,R] = qrappend(Q,R,x) append a column to a QR factorization.
		# If [Q,R] = qr(A) is the original QR factorization of A,
		# then [Q,R] = qrappend(Q,R,x) changes Q and R to be the
		# factorization of the matrix obtained by appending an extra
		# column, x, to A.
		# ported from MATLAB code
		#
		if isempty(Q)
  			fact = qr(x)
  			Q=Matrix(fact.Q)
  			R=fact.R
		else
			~,m = size(Q)
			m += 1
			r = Q'*x			 # best fit of x by Q
			R = [R r] 		 	 # add coeff to R
		  	q= x - Q*r 		 	 # q is orthogonal part of x
			f= norm(q)
		  	R = [R; zeros(1,m);] # update R for q
  			R[m,m] = f[1] 		 # f is coeff of q when normalized
			Q = [Q q/f] 		 # extend basis by normalized q
		end
		#
		#
		return Q,R
	end

	function qrdelete(Q,R,j)
		# qrdelete(Q,R,j) delete the j-th column of A=QR from the QR factorisation
		# That is, Q1*R1 is the matrix Q*R with it's j-th column removed
		# adapted from MATLAB implementation of the same thing
		#
		mq,nq = size(Q)
		R=R[:, 1:end .!=j]
		m,n = size(R)
		#
		for k in j:min(n,m-1)
			p=k:(k+1)
			G,R[p,k] = planerot(R[p,k])
			if k<n
				R[p,k+1:n] = G*R[p,k+1:n]
			end
			Q[:,p]=Q[:,p]*G'
		end
		Q=Q[:,1:n]      #additional clean-up 
		R=R[1:n,1:n]	#bog-standard qrdelete misses this
		#
		#
		return Q,R
	end

	function planerot(x)
		#planerot(x) should do what Givens does - Givens rotation
		#this code, I did just plunder from github
	    if length(x) != 2
	        error()
	    end
	    da, db = x
	    roe = db
	    if abs(da) > abs(db)
	       roe = da
	    end
	    scale = abs(da) + abs(db)
	    if scale == 0
	        c = 1.
	        s = 0.
	        r = 0.
	    else
	        r = scale * sqrt((da/scale)^2 + (db/scale)^2)
	        r = sign(roe) * r
	        c = da/r
	        s = db/r
	    end
	    #
	    #
	    return [c s; -s c], [r, 0.]	
	end

end # of module