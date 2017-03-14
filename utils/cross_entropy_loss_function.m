function ce_loss = cross_entropy_loss_function(y,t)
% <============ HEADER =============>
% @params   : t <- labels
%           y <- outputs
% @returns  : corresponding cross entropy loss
% <============ HEADER =============>

if (size(y) ~= size(t))
    error('Output and labels sizes are not matching');
end

n = size(t,1);
ce_loss = 0;
eps = 0.0001;

for i=1:n
   point_loss = t(i)*log(max(y(i),eps)) + (1-t(i))*log(max(eps,1-y(i)));
   ce_loss = ce_loss - point_loss; 
end

end