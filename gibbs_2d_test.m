% Gibbs Sampler: Test in 2D
% Core algorithm

% restart
close all; clear; clc;

% reset random number generator
rng('default');

% options
Nsteps = 100;
doShowSteps = 0;
doAnimate = 0;

% true PDF parameters
mu_true = [2 5];
sigma_1 = 2;
sigma_2 = 4;
corr = 0.8;
cov_true = [sigma_1^2 corr*sigma_1*sigma_2; corr*sigma_1*sigma_2 sigma_2^2];

% state space to explore (i.e. query points)
xq_vec{1} = (-3:0.1:7)';
xq_vec{2} = (-4:0.1:13)';

% generate PDF at query points
[X1Q,X2Q] = meshgrid(xq_vec{1},xq_vec{2});
XQ = [X1Q(:) X2Q(:)];
[PQ] = mvnpdf(XQ,mu_true,cov_true);
PDF = reshape(PQ,size(X1Q));

% plot true PDF
figure(1);
% set(gcf,'Position',[0821 0195 0419 0783]);
set(gcf,'Position',[0488 1.562000e+02 3.642000e+02 6.058000e+02]);
hold on; grid on;
axis equal;
contour(X1Q,X2Q,PDF);

% start sampler at random query point
x0 = XQ(randi(length(XQ)),:)';
plot(x0(1),x0(2),'.','Color',[0 0.8 0],'MarkerSize',20);

% start history
x = x0;
x_hist = NaN(length(mu_true),Nsteps+1);
x_hist(:,1) = x0;

% show starting point
plot(x0(1),x0(2),'k.','MarkerSize',10);
plot(x0(1),x0(2),'ko','MarkerSize',10,'LineWidth',2.0);

% iterate sampler
for gibbsIter = 1:Nsteps
    for dimIdx = 1:length(mu_true)
        
        % reset storage
        gibbsXQ = repmat(x',length(xq_vec{dimIdx}),1);
        gibbsXQ(:,dimIdx) = xq_vec{dimIdx};
        
        % slice PDF along this dimension at the current point
        gibbsPQ = mvnpdf(gibbsXQ,mu_true,cov_true);
        gibbsPQ = gibbsPQ/trapz(gibbsXQ(:,dimIdx),gibbsPQ);
        
        % compute CDF and sample from it using inverse transform sampling
        gibbsCDF = cumtrapz(gibbsXQ(:,dimIdx),gibbsPQ);
        gibbsPt = gibbsXQ( arrayfun( @(x) find(x <= gibbsCDF,1,'first') , rand(1,1)) ,dimIdx);
        x(dimIdx) = gibbsPt;
        gibbsSamp = gibbsXQ( arrayfun( @(x) find(x <= gibbsCDF,1,'first') , rand(1,100)) ,dimIdx);  % for illustration only
        
        % plot results of stepping along this dimension if requested
        if(doShowSteps)
            figure(1);
            plot(x(1),x(2),'r+','MarkerSize',10,'LineWidth',3);  % selected point
            switch(dimIdx)
                case 1
                    plot(gibbsXQ(:,dimIdx),x(2)*ones(size(gibbsXQ,1),1),'r-','LineWidth',1.0);
                    plot(gibbsSamp,x(2)*ones(size(gibbsSamp)),'r.','MarkerSize',10);  % for illustration only
                case 2
                    plot(x(1)*ones(size(gibbsXQ,1),1),gibbsXQ(:,dimIdx),'r-','LineWidth',1.0);
                    plot(x(1)*ones(size(gibbsSamp)),gibbsSamp,'r.','MarkerSize',10);  % for illustration only
            end
            
            % show 1-D plot
            figure;
            hold on; grid on;
            plot(gibbsXQ(:,dimIdx),gibbsPQ);
            plot(gibbsSamp,0*ones(size(gibbsSamp)),'r.','MarkerSize',10);
            plot(gibbsPt,0,'r+','MarkerSize',10,'LineWidth',3);
        end
        
    end
    
    % store new point in markov chain after moving in each direction
    x_hist(:,gibbsIter+1) = x;
    
    % show result of this iteration if requested
    if(doAnimate)
        figure(1);
        plot(x(1),x(2),'k.','MarkerSize',10);
        plot(x(1),x(2),'ko','MarkerSize',10,'LineWidth',2.0);
        plot(x_hist(1,:),x_hist(2,:),'k.-','MarkerSize',10);
        pause(0.1);
    end
    
end

% show all points in the markov chain
figure(1);
plot(x_hist(1,:),x_hist(2,:),'k.-','MarkerSize',10);
plot(x_hist(1,:),x_hist(2,:),'k.','MarkerSize',10);
plot(x_hist(1,:),x_hist(2,:),'ko','MarkerSize',10,'LineWidth',2.0);