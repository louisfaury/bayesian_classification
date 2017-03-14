function y = compute_output(activation_function, w, input, feature)
% <============ HEADER =============>
% @brief    : computes the output [yn = a(w'*phi(xn))]
% @params   : activation_function <- string for desired activation 
%             w <- current point estimate of learned regressor
%             input 
%             feature <- string for desired feature 
% @returns  : y <- output
% <============ HEADER =============>

n = size(input,1);
y = zeros(n);

switch feature
    case 'linear'
        % do nothing
    otherwise 
        error('Unknwown feature');
end

y = w'*input;

switch activation_function
    case 'logistic_sigmoid'
        arrayfun(@(x) 1/(1+exp(-x)), y);
    otherwise
        error('Unknwown activation function');
        
end
end