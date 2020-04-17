% weighted bootstrap method of sampling from Smith & Gelfand 1992
% have a pdf g(x) and a function f(x) which could (but need not) be normalized to a pdf h(x)
% sample from g(x), then transform those samples so that they could have been generated from h(x)

% restart
close all; clear all; clc;
rng(182,'twister');

% options
doSortParticlesBeforeResamplilng = 0;
doUseStandardBootstrap = 0;

% means and standard deviations of two distributions
mu_g = 3;
sigma_g = 1;
mu_h = 5;
sigma_h = 0.5;

% sample size
% following proof in paper both original sample and resampled set have same size
N = 500;

% domain
x = -1:0.001:8;

% functions g(x), h(x), and f(x)... sampling from g(x), transforming via f(x) to be samples from h(x)
g = (1/(sigma_g*sqrt(2*pi)))*exp(-0.5*((x-mu_g)/sigma_g).^2);
h = (1/(sigma_h*sqrt(2*pi)))*exp(-0.5*((x-mu_h)/sigma_h).^2);  % note: f(x) is NOT normalized...
f = exp(-0.5*((x-mu_h)/sigma_h).^2);  % note: f(x) is NOT normalized...

% sample from g(x)
samp_g = mu_g + randn(N,1)*sigma_g;

% smooth sample from g(x)
% g_ks = ksdensity(samp_g,x,'Kernel','normal');%'Epanechnikov');

% find indices of samples in the x vector
samp_g_idx = arrayfun(@(thisSamp) find(x >= thisSamp,1,'first'),samp_g); 
if( doSortParticlesBeforeResamplilng)
    samp_g_idx = sort(samp_g_idx);
end

% compute weights
if(doUseStandardBootstrap)
    q = (1/N)*ones(1,N);  % standard bootstrap uses a uniform PMF
else
    w = f(samp_g_idx)./g(samp_g_idx); % weight each sample by the ratio of f(x)/g(x) at the appropriate x
    q = w/sum(w); % this is now a PMF (not a PDF)
end
q_cdf = cumsum(q); % the samp_g_idx vector maps these particles back to the samples of g

% resample using inverse transform sampling (i.e. universaility of the uniform)
u = rand(N,1);
rs_sampnums = arrayfun(@(x) find(q_cdf >= x,1,'first'),u);  % sampled particle numbers, not indexed to samples of g(x)
resamp_x = x(samp_g_idx( rs_sampnums ));
resamp_ks = ksdensity(resamp_x,x); %,'Kernel','Epanechnikov');

% plot results
figure;
set(gcf,'Position',[1.540000e+01 4.586000e+02 1.495200e+03 2.888000e+02]);
subplot(1,3,1);
hold on; grid on;
plot(x,g,'-','Color',[0.8 0 0],'LineWidth',1.6);
plot(x,f,'-','Color',[0 0.8 0],'LineWidth',1.6);
plot(x,h,'--','Color',[0 0.8 0],'LineWidth',1.6);
plot(x,resamp_ks,'-','Color',[0 0 0.8],'LineWidth',1.6);
plot(x(samp_g_idx),zeros(1,N),'.','Color',[0.8 0 0],'MarkerSize',10);
plot(resamp_x,zeros(1,N),'o','Color',[0 0 0.8],'MarkerSize',5);
legend('g(x)','f(x)','h(x)','Resample PDF','g(x) Samples','Resample','Location','NorthWest');
xlim([-1 8]);
title('\bfWeighted Bootstrap');
    
subplot(1,3,2);
hold on; grid on;
stem(1:length(q),q,'-','MarkerSize',1,'Color',[0.8 0 0]);
xlabel('\bfOriginal Sample/Particle Index');
ylabel('\bfProbability');
title('\bfSample PMF');

subplot(1,3,3);
hold on; grid on;
plot(1:length(q),q_cdf,'-','Color',[0.8 0 0]);
plot(zeros(size(u)),u,'.','MarkerSize',10,'Color',[0 0 0.8]);
plot( rs_sampnums , zeros(size(rs_sampnums)),'.','MarkerSize',10,'Color',[0.8 0 0.8]);
xlabel('\bfOriginal Sample/Particle Index');
ylabel('\bfProbability');
title('\bfSample CDF and Resample');
ylim([0 1]);
legend('CDF','Uniform Samples','Samples from PMF/CDF','Location','NorthWest');