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
max_iter = 5000;
mini_batch_size = 40;
eps = 1e-4;
feature = 'linear';
learning_rate = 0.002;
loss = 0;
loss_array = zeros(max_iter,1);
cor_hessian = 0.0001;
lambda_l1 = 2;
lambda_l2 = 5;

%% init
w = zeros(input_size+1,1); % zero init

%% run
for iter=1:max_iter
    % minibatch sample and output computation
   [mb, t] = sample_mini_batch(mini_batch_size, dataset, input_size);
   y = compute_output('logistic_sigmoid', w(1:input_size), w(input_size+1), mb, feature);
   
   % update
   R = diag(y.*(1-y));
   switch feature
       case 'linear'
           phi = [mb , ones(size(mb,1),1)];         % bias accounting 
       otherwise 
        error('Unknwown feature')
   end
    
   H = phi'*R*phi + cor_hessian*eye(input_size+1);  % hessian with correction for bad conditioning;
   g = phi'*(y-t);                                  % gradient
   if (nargin>2)                                    % penalisation correction
       switch (opt)
           case 'L1'
               g = g + lambda_l1*sign(w);
           case 'L2'
                H = H + lambda_l2*eye(size(H));
                g = g + lambda_l2*w;
           otherwise
               error('Unknown penalisation keyword');
       end
   end
   d = linsolve(H,g);                               % Newton descent direction
   
   w = w - learning_rate*d;
   
   % convergence check
   ytot = compute_output('logistic_sigmoid', w(1:input_size), w(input_size+1), dataset(:,1:input_size), feature);
   nloss = cross_entropy_loss_function(ytot,dataset(:,input_size+1));
   if (nargin>2)
       switch (opt)
           case 'L1'
               nloss  = nloss + lambda_l1*sum(abs(w));
           case 'L2'
               nloss = nloss + 0.5*lambda_l2*(w'*w);
           otherwise
               error('Unknown penalisation keyword');
       end
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
   if (strcmp(opt,'L1'))
      % sparse formulation 
      w = w.*double(w>0.001);
   end
end
lc = loss_array(1:iter-1);

end
