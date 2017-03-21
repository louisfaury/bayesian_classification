function y = compute_output(activation_function, w, b, input, feature)
% <============ HEADER =============>
% @brief    : computes the output [yn = a(w'*phi(xn))]
% @params   : activation_function <- string for desired activation 
%             w <- current point estimate of learned regressor
%             b <- learned bias 
%             input 
%             feature <- string for desired feature 
% @returns  : y <- output
% <============ HEADER =============>

switch feature
    case 'linear'
        % do nothing
    otherwise 
        error('Unknwown feature');
end

z = input*w + b*ones(size(input,1),1);

switch activation_function
    case 'logistic_sigmoid'
        y = arrayfun(@(x) 1/(1+exp(-x)), z);
    otherwise
        error('Unknwown activation function');
        
end
end