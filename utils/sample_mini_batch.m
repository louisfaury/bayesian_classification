function [mb, t] = sample_mini_batch(mb_size, data, in_size,replace)
% <============ HEADER =============>
% @brief    : sample mini-batch (with or without replacement) from the dataset
% @params   : mb_size <- size of the mini-batch
%             data <- dataset
%             in_size <- input vector dimension 
% @returns  : mb <- sampled mini batched
%             t <- corresponding targets
% <============ HEADER =============>

[mb,ix] = datasample(data(:,1:in_size),mb_size,'Replace',replace);
t = data(ix,in_size+1);

end