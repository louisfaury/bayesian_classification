function [mb, t] = sample_mini_batch(mb_size, data, in_size)
% <============ HEADER =============>
% @brief    : sample mini-batch (with replacement) from the dataset
% @params   : mb_size <- size of the mini-batch
%             data <- dataset
%             in_size <- input vector dimension 
% @returns  : mb <- sampled mini batched
%             t <- corresponding targets
% <============ HEADER =============>

[mb,ix] = datasample(data(:,1:in_size),mb_size);
t = data(ix,in_size+1);

end