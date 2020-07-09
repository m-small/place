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
