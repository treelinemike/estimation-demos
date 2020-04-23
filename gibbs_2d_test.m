% 2D gibbs sampling test

% restart
close all; clear; clc;

% reset random number generator
rng('default');

% options
Nsteps = 100;
doShowSteps = 0;

% true PDF parameters
mu_true = [2 5]';
sigma_1 = 2;
sigma_2 = 4;
corr = 0.8;
cov_true = [sigma_1^2 corr*sigma_1*sigma_2; corr*sigma_1*sigma_2 sigma_2^2];

% state space to explore (i.e. query points)
x1q_vec = -3:0.1:7;
x2q_vec = -4:0.1:13;

% generate PDF at query points
[X1Q,X2Q] = meshgrid(x1q_vec,x2q_vec);
XQ = [X1Q(:), X2Q(:)];
[PQ] = mvnpdf(XQ,mu_true',cov_true);
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
        
        switch dimIdx
            case 1
                % slice PDF along this dimension at the current point
                gibbsx1 = x1q_vec;
                gibbsx2 = x(2)*ones(size(x1q_vec));
                gibbsXQ = [gibbsx1' gibbsx2'];
                gibbsPQ = mvnpdf(gibbsXQ,mu_true',cov_true);
                gibbsPQ = gibbsPQ/trapz(gibbsx1,gibbsPQ);
                
                % compute CDF and sample from it using inverse transform sampling
                gibbsCDF = cumtrapz(gibbsx1,gibbsPQ);
                gibbsPt = gibbsx1(arrayfun( @(x) find(x <= gibbsCDF,1,'first') , rand(1,1) ));
                x(dimIdx) = gibbsPt;
                gibbsSamp = gibbsx1(arrayfun( @(x) find(x <= gibbsCDF,1,'first') , rand(1,100) ));  % for illustration only
                
                if(doShowSteps)
                    figure(1);
                    plot(gibbsx1,gibbsx2,'r-','LineWidth',1.0);
                    plot(x(1),x(2),'r+','MarkerSize',10,'LineWidth',3);
                    plot(gibbsSamp,x(2)*ones(size(gibbsSamp)),'r.','MarkerSize',10);  % for illustration only
                    
                    figure;
                    hold on; grid on;
                    plot(gibbsx1,gibbsPQ);
                    plot(gibbsSamp,0*ones(size(gibbsSamp)),'r.','MarkerSize',10);
                    plot(gibbsPt,0,'r+','MarkerSize',10,'LineWidth',3);
                end
                
            case 2
                % slice PDF along this dimension at the current point
                gibbsx1 = x(1)*ones(size(x2q_vec));
                gibbsx2 = x2q_vec;;
                gibbsXQ = [gibbsx1' gibbsx2'];
                gibbsPQ = mvnpdf(gibbsXQ,mu_true',cov_true);
                gibbsPQ = gibbsPQ/trapz(gibbsx2,gibbsPQ);
                
                % compute CDF and sample from it using inverse transform sampling
                gibbsCDF = cumtrapz(gibbsx2,gibbsPQ);
                gibbsPt = gibbsx2(arrayfun( @(x) find(x <= gibbsCDF,1,'first') , rand(1,1) ));
                x(dimIdx) = gibbsPt;
                gibbsSamp = gibbsx2(arrayfun( @(x) find(x <= gibbsCDF,1,'first') , rand(1,100) ));  % for illustration only
                
                if(doShowSteps)
                    figure(1);
                    plot(gibbsx1,gibbsx2,'b-','LineWidth',1.0);
                    plot(x(1),x(2),'b+','MarkerSize',10,'LineWidth',3);
                    plot(x(1)*ones(size(gibbsSamp)),gibbsSamp,'b.','MarkerSize',10);  % for illustration only
                    
                    figure;
                    hold on; grid on;
                    plot(gibbsx2,gibbsPQ);
                    plot(gibbsSamp,0*ones(size(gibbsSamp)),'b.','MarkerSize',10);
                    plot(gibbsPt,0,'b+','MarkerSize',10,'LineWidth',3);
                end
                
        end
        
        
    end
    x_hist(:,gibbsIter+1) = x;
    figure(1);
    plot(x(1),x(2),'k.','MarkerSize',10);
    plot(x(1),x(2),'ko','MarkerSize',10,'LineWidth',2.0);
    plot(x_hist(1,:),x_hist(2,:),'k-');
    
end



