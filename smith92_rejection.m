% restart
close all; clear all; clc;
rng(182,'twister');

% means and standard deviations of two distributions
mu_g = 3;
sigma_g = 1;
mu_h = 5;
sigma_h = 0.5;

% number of samples
N = 50000

% constant factor
M = 30; 

% domain
x = 0:0.001:10;

g = (1/(sigma_g*sqrt(2*pi)))*exp(-0.5*((x-mu_g)/sigma_g).^2);
h = (1/(sigma_h*sqrt(2*pi)))*exp(-0.5*((x-mu_h)/sigma_h).^2);


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
   if( u(sampIdx) <=  h(thisXIdx)/(g(thisXIdx)*M) )
       samp_h(end+1) = samp_g(sampIdx);
   end
end


% plot results
figure;
hold on; grid on;
plot(x,g,'--','LineWidth',1.6,'Color',[0 0.6 0]);
plot(x,h,'--','LineWidth',1.6,'Color',[0.6 0 0]);
plot(samp_g,zeros(size(samp_g)),'.','MarkerSize',10,'Color',[0 1 0]);
plot(x,g_ks,'LineWidth',1.6,'Color',[0 1 0]);
% plot(x,h./g,'m-','LineWidth',1.6);
plot(samp_h,zeros(size(samp_h)),'.','MarkerSize',5,'Color',[1 0 0]);
h_ks = ksdensity(samp_h,x,'Kernel','normal');%'Epanechnikov');
plot(x,h_ks,'LineWidth',1.6,'Color',[1 0 0]);

% show stats 
N_exp = ceil((1/M)*trapz(x,h)*N);
fprintf('%d prior particles downsampled to %d posterior particles (expected %d).\n',length(samp_g),length(samp_h),N_exp);
 