% test kernel smoothing functions that produce PDF from particles
% 1. ksdensity() in MATLAB (Gaussian and Epanechnikov kernels)
% 2. "Regularized particle filter resampling" (Epanechnikov) from pp473-4 of Simon2006 "Optimal State Estimation"

% restart
close all; clear all; clc;

% options
mu_true = 0.453;
sigma_true = 0.04;
Np = 2000;
x_test = -2:0.001:2;

% generate true PDF
true_pdf = normpdf(x_test,mu_true,sigma_true);

% sample "particles" from true PDF
rng('default');  % make sure we draw the same random particles every time
x_part = mu_true + sigma_true*randn(1,Np);

% compute density using a Gaussian kernel from MATLAB
ks_pdf_a = ksdensity(x_part,x_test);

% compute density using a Epanechnikov (parabolic) kernel from MATLAB
[ks_pdf_b,~,matlabBW] = ksdensity(x_part,x_test,'Kernel','epanechnikov');

% compute density using Epanechnikov (parabolic) kernel (Dan Simon, p. 473-474)
qi = (1/Np)*ones(1,Np); % flat prior, not ideal? ... these are the a priori weights....
ks_pdf_c = part2pdf(x_part,qi,x_test);

figure;
set(gcf,'Position',[-056 0558 1930 0267]);
hold on; grid on;
plot(x_test,true_pdf,':','Color',[ 0 0.8 0 ],'LineWidth',1.6);
plot(x_test,ks_pdf_a,'-','Color',[ 0.8 0 0 ],'LineWidth',1.6);
plot(x_test,ks_pdf_b,'-','Color',[ 0 0 0.8 ],'LineWidth',1.6);
plot(x_test,ks_pdf_c,'-','Color',[ 0.8 0 0.8 ],'LineWidth',1.6);

% fprintf('Matlab BW: %8.6f\n',matlabBW);
fprintf('MATLAB / Simon PDF RMSE: %8.6f\n',rms(ks_pdf_c-ks_pdf_b));