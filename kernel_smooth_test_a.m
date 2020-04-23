% test kernel smoothing functions that produce PDF from particles
% 1. ksdensity() in MATLAB (Gaussian and Epanechnikov kernels)
% 2. "Regularized particle filter resampling" (Epanechnikov) from pp473-4 of Simon2006 "Optimal State Estimation"

% restart
close all; clear all; clc;

% options
doUseTwoModes = 1;

% true density parameters
mu_1_true = 0.453;
sigma_1_true = 0.04;
mu_2_true = -0.790;
sigma_2_true = 0.08;

Np = 8000;
x_test = -2:0.001:2;

% generate true PDF
if(doUseTwoModes)
    true_pdf = normpdf(x_test,mu_1_true,sigma_1_true)/2 + normpdf(x_test,mu_2_true,sigma_2_true)/2;
else
    true_pdf = normpdf(x_test,mu_1_true,sigma_1_true);    
end
% sample "particles" from true PDF
rng('default');  % make sure we draw the same random particles every time

true_cdf = cumtrapz(x_test,true_pdf);
x_part = x_test(arrayfun( @(x) find(x <= true_cdf,1,'first') , rand(1,Np) ));
% x_part = mu_1_true + sigma_1_true*randn(1,Np);

% compute density using a Gaussian kernel from MATLAB
ks_pdf_a = ksdensity(x_part,x_test);

% compute density using a Epanechnikov (parabolic) kernel from MATLAB
[ks_pdf_b,~,matlabBW] = ksdensity(x_part,x_test,'Kernel','epanechnikov');

% compute density using Epanechnikov (parabolic) kernel (Dan Simon, p.
% 473-474) WITH BANDWIDTH SCALED TO TYPICAL OPTIMAL BW WHICH WILL MATCH
% MATLAB CLOSELY
qi = (1/Np)*ones(1,Np); % flat prior, ideal case for epanechnikov? ... these are the weights describing the likelihood of the a priori particles given the observation....
ks_pdf_c = part2pdf(x_part,qi,x_test,2.0);

% Epanechnikov via Simon approach with BANDWIDTH SCALED PER SIMON FOR BEST
% ACCOMMODATION OF MULTIMODAL PDFs
ks_pdf_d = part2pdf(x_part,qi,x_test,1.0);


figure;
set(gcf,'Position',[-056 0558 1930 0267]);
hold on; grid on;
plot(x_test,true_pdf,':','Color',[ 0 0.8 0 ],'LineWidth',1.6);
plot(x_test,ks_pdf_a,'-','Color',[ 0.8 0 0 ],'LineWidth',1.6);
plot(x_test,ks_pdf_b,'-','Color',[ 0 0 0.8 ],'LineWidth',1.6);
plot(x_test,ks_pdf_c,'-','Color',[ 0.8 0 0.8 ],'LineWidth',1.6);
plot(x_test,ks_pdf_d,'-','Color',[ 0.8 0.8 0 ],'LineWidth',1.6);
plot(x_part,zeros(size(x_part)),'k.','MarkerSize',10);
legend('True PDF','ksdensity() Gaussian','ksdensity() Epanechnikov','Simon Epanechnikov bw*2.0','Simon Epanechnikov bw*1.0','Samples');

% scale plot
xrange = max(x_part)-min(x_part);
rangescale = 0.125;
xlim( rangescale*xrange*[-1 1] + [min(x_part) max(x_part)]);

% fprintf('Matlab BW: %8.6f\n',matlabBW);
fprintf('MATLAB / Simon PDF RMSE: %8.6f\n',rms(ks_pdf_c-ks_pdf_b));