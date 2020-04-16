% rejection method of sampling from Smith & Gelfand 1992
% have a pdf g(x) and a function f(x) which could (but need not) be normalized to a pdf h(x)
% sample from g(x), then transform those samples so that they could have been generated from h(x)

% restart
close all; clear all; clc;
rng(182,'twister');

% means and standard deviations of two distributions
mu_g = 3;
sigma_g = 1;
mu_h = 5;
sigma_h = 0.5;

% sample size
N = 50000;

% constant factor
M = 30; 

% domain
x = -1:0.001:8;

% functions g(x) and h(x)... sampling from g(x), transforming to be samples from h(x)
g = (1/(sigma_g*sqrt(2*pi)))*exp(-0.5*((x-mu_g)/sigma_g).^2);
h = (1/(sigma_h*sqrt(2*pi)))*exp(-0.5*((x-mu_h)/sigma_h).^2);  % note: f(x) is NOT normalized...
f = exp(-0.5*((x-mu_h)/sigma_h).^2);  % note: f(x) is NOT normalized...

% sample from g(x)
samp_g = mu_g + randn(N,1)*sigma_g;

% smooth sample from g(x)
g_ks = ksdensity(samp_g,x,'Kernel','normal');%'Epanechnikov');

% sample from uniform distribution
u = rand(size(samp_g));

% accept/reject samples
samp_h = [];
for sampIdx = 1:length(samp_g)
   thisXIdx = find( x >= samp_g(sampIdx),1,'first');
   if( u(sampIdx) <=  f(thisXIdx)/(g(thisXIdx)*M) )
       samp_h(end+1) = samp_g(sampIdx);
   end
end

% smooth accepted samples
h_ks = ksdensity(samp_h,x);%,'Kernel','Epanechnikov');

% plot results
figure;
set(gcf,'Position',[0085 0309 1.220800e+03 0420]);
hold on; grid on;
plot(x,g,'-','LineWidth',1.6,'Color',[0.8 0 0]);
plot(x,f,'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot(x,h,'--','LineWidth',1.6,'Color',[0 0.8 0]);
% plot(x,g_ks,':','LineWidth',1.6,'Color',[0.8 0 0]);
% plot(x,h./g,'m-','LineWidth',1.6);
plot(x,h_ks,'LineWidth',1.6,'Color',[0 0 0.8]);
plot(samp_g,zeros(size(samp_g)),'.','MarkerSize',10,'Color',[0.8 0 0]);
plot(samp_h,zeros(size(samp_h)),'o','MarkerSize',5,'Color',[0 0 0.8]);
legend('g(x)','f(x)','h(x)','Accepted PDF','g(x) Samples','Accepted','Location','NorthWest');
xlim([-1 8]);
title('\bfRejection Method');

% show stats 
N_exp = ceil((1/M)*trapz(x,h)*N);
fprintf('%d prior particles downsampled to %d posterior particles (expected %d).\n',length(samp_g),length(samp_h),N_exp);
 