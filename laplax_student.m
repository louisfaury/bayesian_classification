function [wMap,Sn] = laplax_student(ds, is, prior,plotflag)
% <============ HEADER =============>
% @brief    : computes the posterior approximation with Laplace
%             approximation, assuming Student prior 
%             feature space is input space 
% @params   : ds <- dataset
%             is <- input vectors dimensionality
%             prior <- {mean,covariance matrix} structure -- assuming
%                       Gaussian prior for now 
%             plotflag <- boolean flag for plot
% @returns  : posterior distribution q
% <============ HEADER =============>

t       = ds(:,is+1);               % targets
nu      = prior.nu;                 % student prior 
outp    = @(w) compute_output('logistic_sigmoid',w(1:is),w(is+1),ds(:,1:is),'linear');
obj     = @(w) cross_entropy_loss_function(outp(w),t) - log(prod(tpdf(w,nu)));
n       = is+1;

% Finding the posterior MAP value
max_iter    = 20;
lr          = 0.01;
w           = zeros(is+1,1);
obj_array   = zeros(max_iter,1);
eps = 0.001;
for i=1:max_iter
   X = [ds(:,1:is),ones(size(ds,1),1)]; y = outp(w);    % design - pred - labels 
   g = (w.*(nu+1))./(nu+w.^2) -X'*(t-y);                % gradient
   d = g;
   w = w - lr*d;
   if (abs(lr*d)<eps)
       break;                                           % stopping criterion 
   end
   lr = lr*0.99;
   obj_array(i) = obj(w);
end

if (nargin>3 && plotflag)
    plot(obj_array,'b','LineWidth',2);
    xlabel('Iterations');
    ylabel('$\log(p(\theta\,\vert\, X))$','interpreter','latex','FontSize',12);
    title('Laplace Log Posterior Minimization (Student Prior)');
end

wMap = w; 
Sn = inv(X'*diag(y.*(1-y))*X + (1+nu).*(nu-w.^2)./((w.^2+nu).^2));
end