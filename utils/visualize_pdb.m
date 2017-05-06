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
figure('units','normalized','outerposition',[0 0 1 1]); hold on;
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

x1 = [min(proj_data(:,1))-1:0.1:max(proj_data(:,1))+1];
x2 = [min(proj_data(:,2))-1:0.1:max(proj_data(:,2))+1];
[X,Y] = meshgrid(x1,x2);
for i=1:size(x2,2)
    for j=1:size(x1,2)
        Z(i,j) = pred_db_2d(X(i,j),Y(i,j),proj_w,proj_S);
    end
end
map = ([ 0.1:0.01:0.9; 0.9:-0.01:0.1; 0.2*ones(1,81)])';
subplot(1,2,1); hold on;
contourf(X,Y,Z,40,'LineStyle','none'); hold on; contour(X,Y,Z,[0.05,0.25, 0.5, 0.75,0.95],'Showtext','on','LineColor','black'); colormap(map); colorbar; hold on; 

hold on;
a = scatter(proj_data(ds(:,is+1)==1,1),proj_data(ds(:,is+1)==1,2),'MarkerFaceColor',[1, 0.1, 0.2],'MarkerEdgeColor','black');
b = scatter(proj_data(ds(:,is+1)==0,1),proj_data(ds(:,is+1)==0,2),'MarkerFaceColor',[0.1, 1, 0.2],'MarkerEdgeColor','black');
legend([a,b],'True Label : 1','True Label : 0','Location','northwest');

% Donut plot 
subplot(1,2,2); hold on;
pred_ppoint = pred_db(data,w,S);
colormap(map);
scatter(proj_data(:,1),proj_data(:,2),200*ones(size(proj_data,1),1),(pred_ppoint-min(pred_ppoint))/(max(pred_ppoint)-min(pred_ppoint)),'filled','MarkerEdgeColor','white');
scatter(proj_data(ds(:,is+1)==1,1),proj_data(ds(:,is+1)==1,2),30*ones(size(proj_data(ds(:,is+1)==1,1))),'MarkerFaceColor',[1, 0.1, 0.2],'MarkerEdgeColor','white','LineWidth',2);
scatter(proj_data(ds(:,is+1)==0,1),proj_data(ds(:,is+1)==0,2),30*ones(size(proj_data(ds(:,is+1)==0,1))),'MarkerFaceColor',[0.1, 1, 0.2],'MarkerEdgeColor','white','LineWidth',2);

end
