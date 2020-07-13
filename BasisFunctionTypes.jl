function tophat(x,p)   #first function evaluates the vector input x at parameter p
    return exp.(-x.^p)
end
function dtophat(x,p)  #second function is the derivative
	return -p* (x.^(p-1)) .* exp.(-x.^p)
end
function tophat() #third function, with no input arguments, gives the default range of parameters p
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

function quintic(x)
	return x.^5
end
function quintic()
	return []
end
function dquintic(x)
	return 5 .* x.^4
end

function wavelet(x)
	return (2*x.^2-1) .* exp.(-x.^2)
end
function wavetlet()
	return []
end
function dwavelet(x)
	return (2*x.*(3-2*x.^2)) .* exp.(-x.^2)
end

function sigmoid(x)
	return tanh.( x )
end
function sigmoid()
	return []
end
function dsigmoid(x)
	return (1-tanh.( x ).^2)
end

function morlet(x,p)
	return cos.(2*pi*x) .* exp.(-(2*(pi/p)^2)*x.^2) .- exp.(-p^2/2-(2*(pi/p)^2)*x.^2)
end
function morlet()
	return [1 9;]
end
function dmorlet(x,p)
	return -2*(pi.*sin(2*pi*x) .+ 2*((pi/p)^2)*x.*cos.(2*pi*x)) .* exp(-(2*(pi/p)^2)*x.^2) .+ (4*((pi/p)^2)*x) .* exp.(-p^2/2-(2*(pi/p)^2)*x.^2)
end





