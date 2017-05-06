function pdb = pred_db(ds,w,S)
% <============ HEADER =============>
% @brief    : computes the predictive distribution (full dataset)
% @param    : ds <- dataset
%             (w,S) <- Gaussian based posterior approximation 
% <============ HEADER =============>
n = size(ds,1);

sig     = @(x) 1/(1+exp(-x));
pdbf    = @(x) sig((w'*x)/(1+pi*(x'*S*x)^(2/8))^(1/2)); % see Bishop, chapter on Bayesian Binary Classification 
pdb     = zeros(size(ds,1),1);
feat    = [ds,ones(n,1)];
for i=1:n
   pdb(i) = pdbf(feat(i,:)') ;
end

end