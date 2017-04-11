function p = gaussianDb(n, mu, Sigma)
    p = @(x) 1/(2*pi*det(Sigma))^(n/2)*exp(-0.5*(x-mu)'*inv(Sigma)*(x-mu));
end