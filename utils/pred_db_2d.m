function pdb = pred_db_2d(x,y,w,S)
% <============ HEADER =============>
% @brief    : computes the predictive distribution (2d) 
% @param    : ds <- dataset
%             (w,S) <- Gaussian based posterior approximation 
%             is   <- input dimension      
% <============ HEADER =============>

%%
feat = [x;y];
mu = w'*feat;
sigma = feat'*S*feat;
sig = @(x) 1/(1+exp(-x));
pdb = sig((mu/(1+pi*sigma^2/8)^(1/2)));

end