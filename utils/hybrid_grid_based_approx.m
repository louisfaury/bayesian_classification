function [E,Sigma] = hybrid_grid_based_approx(mu,nu,ti)

ti = 2*ti-1;
% Defines hybrid 
sig = @(x) 1/(1+exp(-x));
h   = @(w) sig(ti*w) * exp(-0.5*(w-mu)^2/nu);

% MAP approximation for center of the grid 
max_iter    = 50; 
w           = mu;
lr          = 0.05;
log_hybrid  = zeros(max_iter,1);
eps = 0.00001;
for iter=1:max_iter
    d = ti*(1-sig(ti*w))-(w-mu)/nu;
    w = w + lr*d;
    log_hybrid(iter) = log(h(w));
    if (iter>1) && abs(log_hybrid(iter,1) - log_hybrid(iter-1,1))<eps
        break;
    end
end

% Compute grid 
grid_size = 50;
bound_grid = 10*sqrt(1 / ( 1/nu + sig(ti*w)*(1-sig(ti*w)) ));
grid = linspace(w-bound_grid/2,w+bound_grid/2,grid_size)';
h_on_grid = arrayfun(h,grid);
size = bound_grid/grid_size;
h_on_grid = h_on_grid / (sum(h_on_grid)*size);
% Grid based
mean_grid = h_on_grid'*grid*size;
var_grid = ((grid-mean_grid).^2)'*h_on_grid*size;
% Laplace approximator (deprecated)
% wmap = w;
% Sigma = 1 / ( 1/nu + sig(ti*w)*(1-sig(ti*w)) );
% Attribution
E = mean_grid;
Sigma = var_grid; 

end