function ce_loss = cross_entropy_loss_function(y,t)
% <============ HEADER =============>
% @params   : t <- labels
%           y <- outputs
% @returns  : corresponding cross entropy loss
% <============ HEADER =============>

if (size(y,1) ~= size(t,1))
    error('Output and labels sizes are not matching');
end

n = size(t,1);
ce_loss = 0;

for i=1:n
   point_loss = t(i)*log(y(i)) + (1-t(i))*log(1-y(i));
   ce_loss = ce_loss + point_loss; 
end

end