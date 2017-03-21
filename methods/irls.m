function [w, lc] = irls(dataset, input_size, opt)
% <============ HEADER =============>
% @brief    : computes the ML solution for the logistic regression program.
%             feature space is input space 
% @params   : dataset
%             input_size <- input vectors dimensionality
%             opt <- penalization option : '', 'L1' or 'L2'
% @returns  : learned vector for prediction
%             learning curve
% <============ HEADER =============>

%% algo parameters
max_iter = 4000;
mini_batch_size = 40;
eps = 5e-7;
feature = 'linear';
learning_rate = 0.001;
loss = 0;
loss_array = zeros(max_iter,1);
cor_hessian = 0.0001;

%% init
w = zeros(input_size,1); % zero init

%% run
for iter=1:max_iter
    % minibatch sample and output computation
   [mb, t] = sample_mini_batch(mini_batch_size, dataset, input_size);
   y = compute_output('logistic_sigmoid', w, mb, feature);
   
   % update
   R = diag(y.*(1-y));
   switch feature
       case 'linear'
           phi = mb;
       otherwise 
        error('Unknwown feature')
   end
   
   H = phi'*R*phi + cor_hessian*eye(input_size);    % hessian with correction for bad conditioning;
   g = phi'*(y-t);                                  % gradient
   d = linsolve(H,g);                               % Newton descent direction
   
   w = w - learning_rate*d;
   
   % convergence check
   ytot = compute_output('logistic_sigmoid', w, dataset(:,1:input_size), feature);
   nloss = cross_entropy_loss_function(ytot,dataset(:,input_size+1));

   if (iter>1)
       if (abs(nloss-loss)<eps)
           break;
       end
   end
   
   loss = nloss;
   loss_array(iter,1) = loss;
end

lc = loss_array(1:iter-1);

end
