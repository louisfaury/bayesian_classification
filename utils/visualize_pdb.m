function visualize_pdb(ds,w,S,is)
% <============ HEADER =============>
% @function : visualize_pdb (predictive distribution)
% @brief    : Enables visualization using the predictive distribution 
%             using the logit approximation + plot decision frontiers 
% @param    : ds <- dataset
%             (w,S) <- Gaussian based posterior approximation 
%             is   <- input dimension      
% <============ HEADER =============>

%% TODO :
% grid representation of the world (fine mesh)
% predict the predictive distribution on every point (see Bishop)
% come up with decision criteria (based on the covariance, trust in the
% guess ? something like w+std -> other prediction ? 
% plot the whole (might take some time)

% projection rule 
data = ds(:,1:is);
data = data ./ sqrt(var(data));
empirical_cov = (data-mean(data))'*(data-mean(data));
[V,~] = eigs(empirical_cov,is,'sm');
retained_dim = 2;
proj_matrix = V(:,1:retained_dim)';
proj_data = (data-mean(data))*proj_matrix';
proj_w = proj_matrix*w(1:is);
proj_S = proj_matrix*S(1:is,1:is)*proj_matrix';

x1 = [-10:0.1:10];
x2 = [-10:0.1:10];
[X,Y] = meshgrid(x1,x2);
for i=1:size(x1,2)
    for j=1:size(x2,2)
        Z(i,j) = pred_db_2d(X(i,j),Y(i,j),proj_w,proj_S);
    end
end
map = ([ 0.1:0.01:0.9; 0.9:-0.01:0.1; 0.2*ones(1,81)])';
contourf(X,Y,Z,40,'LineStyle','none'); hold on; contour(X,Y,Z,[0.05,0.25, 0.5, 0.75,0.95],'Showtext','on','LineColor','black'); colormap(map); colorbar; hold on; 

hold on;
for i=1:size(data,1)
    if (ds(i,is+1))
        color = [1, 0.1, 0.2];
         m = plot(proj_data(i,1),proj_data(i,2),'o','MarkerSize',5,'MarkerFaceColor',color,'MarkerEdgeColor','black');
    else
        color = [0.1, 1, 0.2];
        b = plot(proj_data(i,1),proj_data(i,2),'o','MarkerSize',5,'MarkerFaceColor',color,'MarkerEdgeColor','black');
    end
end

end
