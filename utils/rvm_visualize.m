function rvm_visualize(ds,is,width)
% <============ HEADER =============>
% @brief    : plot the decision borders of the RVM 
% @params   : w  <- infered predictor
%             ds <- dataset
%             is <- input vectors dimensionality
%             width <- RBF kernel width
% @returns  : 
% <============ HEADER =============>
figure('units','normalized','outerposition',[0 0 1 1]); hold on;

% projection rule
n = size(ds,1);
data = ds(:,1:is);
data = data ./ sqrt(var(data));
empirical_cov = (data-mean(data))'*(data-mean(data));
[V,~] = eigs(empirical_cov,is,'sm');
retained_dim = 2;
proj_matrix = V(:,1:retained_dim)';
proj_data = (data-mean(data))*proj_matrix';
% Trains the RVM on the projected data 
w = rvm_train([proj_data,ds(:,is+1)],2,width);

x1 = [min(proj_data(:,1))-1:0.1:max(proj_data(:,1))+1];
x2 = [min(proj_data(:,2))-1:0.1:max(proj_data(:,2))+1];
[X,Y] = meshgrid(x1,x2);
for i=1:size(x2,2)
    for j=1:size(x1,2)
        Z(i,j) = rvm_pred(proj_data,X(i,j),Y(i,j),w,width);
    end
end
map = ([ 0.1:0.01:0.9; 0.9:-0.01:0.1; 0.2*ones(1,81)])';
subplot(1,2,1); hold on;
contourf(X,Y,Z,40,'LineStyle','none'); hold on; contour(X,Y,Z,[0.25,0.5,0.75],'Showtext','on','LineColor','black'); colormap(map); colorbar; hold on; 

hold on;
a = scatter(proj_data(ds(:,is+1)==1,1),proj_data(ds(:,is+1)==1,2),'MarkerFaceColor',[1, 0.1, 0.2],'MarkerEdgeColor','black');
b = scatter(proj_data(ds(:,is+1)==0,1),proj_data(ds(:,is+1)==0,2),'MarkerFaceColor',[0.1, 1, 0.2],'MarkerEdgeColor','black');
c = scatter(proj_data(w(1:n)~=0,1),proj_data(w(1:n)~=0,2),'MarkerEdgeColor',[0.1,0.3,0.5],'LineWidth',1.5);
legend([a,b,c],'True Label : 1','True Label : 0','Relevant Vector','Location','northwest');

end


function y = rvm_pred(inputs,x1,y1,w,width)
x = [x1 y1];
BASIS	= exp(-distSquared(inputs,x)/(width^2));
BASIS = [BASIS;1];
sig = @(x) 1/(1+exp(-x));
y = arrayfun(sig,w'*BASIS);
end

function D2 = distSquared(X,Y)
%
nx	= size(X,1);
ny	= size(Y,1);
%
D2 = (sum((X.^2), 2) * ones(1,ny)) + (ones(nx, 1) * sum((Y.^2),2)') - ...
     2*X*Y';
end