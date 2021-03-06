function [w, prior, lc] = irls(dataset, input_size, opt)
% <============ HEADER =============>
% @brief    : computes the ML solution for the logistic regression program.
%             feature space is input space 
% @params   : dataset
%             input_size <- input vectors dimensionality
%             opt <- penalization option : '', 'L1' or 'L2'
% @returns  : learned vector for prediction
%             learned prior for the '+1' class 
%             learning curve
% <============ HEADER =============>

%% algo parameters
max_iter = 50;
mini_batch_size = size(dataset,1);
eps = 1e-2;
feature = 'linear';
learning_rate = 0.2;
loss = 0;
loss_array = zeros(max_iter,1);
lambda = opt.hp;

%% init
w = zeros(input_size+1,1);

%% run
for iter=1:max_iter
    % minibatch sample and output computation
   [mb, t] = sample_mini_batch(mini_batch_size, dataset, input_size,false);
   y = compute_output('logistic_sigmoid', w(1:input_size), w(input_size+1), mb, feature);
   
   % update
   R = diag(y.*(1-y));
   switch feature
       case 'linear'
           phi = [mb , ones(size(mb,1),1)];         % bias accounting 
       otherwise 
        error('Unknwown feature')
   end
    
   H = phi'*R*phi;  % hessian with correction for bad conditioning;
   g = phi'*(y-t);                                  % gradient
   switch (opt.name)
       case 'L1'
           g = g + lambda*sign(w);
       case 'L2'
           H = H + lambda*eye(size(H));
           g = g + lambda*w;
   end
   d = linsolve(H,g);                               % Newton descent direction
   
   w = w - learning_rate*d;
   
   % convergence check
   ytot = compute_output('logistic_sigmoid', w(1:input_size), w(input_size+1), dataset(:,1:input_size), feature);
   nloss = cross_entropy_loss_function(ytot,dataset(:,input_size+1));
   switch (opt.name)
       case 'L1'
           nloss  = nloss + lambda*sum(abs(w(1:input_size)));
       case 'L2'
           nloss = nloss + 0.5*lambda*(w(1:input_size)'*w(1:input_size));
   end
   
   if (iter>1)
       if (abs(nloss-loss)<eps)
           break;
       end
   end
   
   loss = nloss;
   loss_array(iter,1) = loss;
end

if (nargin>2)
   if (strcmp(opt.name,'L1'))
      % sparse formulation 
      w = w.*double(abs(w)>0.001);
   end
end
lc = loss_array(1:iter-1);

%% learning priors (naive)
prior = sum(dataset(:,input_size+1))/size(dataset,1);

end
