% restart
close all; clear; clc;
rng(1123,'twister');

% options
doMakeVideo = 1;

% true parameters
mu = 0;
sigma = 3;
% N_samp_vals = round(logspace(log10(10),log10(10000),10));
N_samp_vals = [2 5 10 20 50 100 200 500 1000 2000];
% state space to explore
x_qp = -12:0.01:12';

% initialize figure
figure;
hold on; grid on;

% smooth for different numbers of samples
for nIdx = 1:length(N_samp_vals)
    
    N_samp = N_samp_vals(nIdx);
    
    d_qp = normpdf(x_qp,mu,sigma);
    
    x_samp = mu + sigma*randn(1,N_samp);
    
    
    d = 1;   % dimensionality of state space
    sig_direct = std(x_samp);
    sig_matlab = 6.6*mad(x_samp',1,1);  % approximately what is used in mvksdensity, see code and try empirically
    sig_matlab/sig_direct
    
    h=sig_matlab*(4/((1+2)*N_samp))^(1/(1+4));
    
    
    
    quad_params = [(h/2)^2 -(h/2) 1; (h/2)^2 (h/2) 1; (2/3)*(h/2)^3 0 2*(h/2)]\[0; 0; 1]
    
    
    cla;
    plot(x_qp,d_qp,'-','LineWidth',1.6,'Color',[0 0 0.8]);
    
    ks_pdf = zeros(size(x_qp));
    for sampIdx = 1:N_samp
        ks_comp =  (1/N_samp)*(quad_params(1).*(x_qp-x_samp(sampIdx)).^2 + quad_params(2).*(x_qp-x_samp(sampIdx)) + quad_params(3));
        ks_comp(x_qp > x_samp(sampIdx)+(h/2)) = 0;
        ks_comp(x_qp < x_samp(sampIdx)-(h/2)) = 0;
        plot(x_qp,ks_comp,'-','LineWidth',1.6,'Color',[0 0.8 0]);
        ks_pdf = ks_pdf + ks_comp;
    end
    
    
    plot(x_qp,ks_pdf,'-','LineWidth',1.6,'Color',[1 0 1]);
    
    
    plot(x_samp,zeros(size(x_samp)),'.','MarkerSize',15,'Color',[0.8 0 0]);
    
    
    xlim([min(x_qp) max(x_qp)]);
    ylim([0 1.2*max(d_qp)]);
    title(sprintf('\\bfSmoothing with N = %4d Samples',N_samp));
    drawnow;
    
    if(doMakeVideo)
        thisImgFile = sprintf('frame%03d.png',nIdx);
        saveas(gcf,thisImgFile);
        system(['convert -trim ' thisImgFile ' ' thisImgFile]);  % REQUIRES convert FROM IMAGEMAGICK!
    else
        pause(0.1);
    end
    
end

if(doMakeVideo)
    system(['ffmpeg -y -r 0.8 -start_number 1 -i frame%03d.png -vf "format=rgba,scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 25 -r 25 output.mp4']);
    system('del frame*.png');
end

% ks_pdf_matlab = ksdensity(x_samp,x_qp,'kernel','Epanechnikov');
% plot(x_qp,ks_pdf_matlab,'--','LineWidth',1.6,'Color',[1 0 1]);

