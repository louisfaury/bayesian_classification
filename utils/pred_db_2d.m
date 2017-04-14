function pdb = pred_db_2d(x,y,w,S)
%% TODO header

%%
feat = [x;y];
mu = w'*feat;
sigma = feat'*S*feat;
sig = @(x) 1/(1+exp(-x));
pdb = sig((mu/(1+pi*sigma^2/8)^(1/2)));

end