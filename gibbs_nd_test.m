% Gibbs Sampler: Test in n-dimensions (two, but extensible)
% First use Epaneschnikov kernel denisty smoothing to reconstruct PDF from samples

% restart
close all; clear; clc;

% reset random number generator
rng('default');

% options
doUseExternalGibbsFcn = 1;
doOneStepOnly = 0;             % does nothing if using exernal Gibbs function
doShowSteps = 0;               % does nothing if using exernal Gibbs function
doAnimate = 0;                 % does nothing if using exernal Gibbs function
doMakeVideo = 0;               % only if animation enabled and not using external Gibbs
stepsToAnimate = 50;           % if animating only show this many steps in Markov chain
Np = 2000;                     % number of particles to sample from the true PDF
hss = 0.1;                     % step size in state space TODO: allow different step sizes in each dimension
NSD = 4;                       % number of standard deviations to extend query points (+/-) in each dimension from mean
bwScale = 1.0;                 % multiply optimal bandwidth by this factor

% Gibbs Sampler Options
gibbsBurnIn = 500;             % discard this many samples before capturing points
gibbsM = 10;                  % after burn in, capture every m-th sample

% true PDF parameters
mu_true = [2 5]';
sigma_1 = 2;
sigma_2 = 4;
corr = 0.8;
cov_true = [sigma_1^2 corr*sigma_1*sigma_2; corr*sigma_1*sigma_2 sigma_2^2];

% sample from the true PDF
x_samp_pre = mvnrnd(mu_true',cov_true,Np)';

% state space to explore (i.e. query points)
nDim = size(x_samp_pre,1);

% principal axes of initial particle set
mu = mean(x_samp_pre,2);
cov = 1/(size(x_samp_pre,2)-1)*(x_samp_pre-mu)*(x_samp_pre-mu)';
[vec,val] = eig(cov);
acos(vec(1,1))*180/pi
% make sure that eigenvector matrix gives a right-handed coordinate system
% (in n-dimensions)... need det(vec) = 1.0 not -1.0
% TODO: may need to further test and refine this!
if(det(vec) < 0)
    vec(:,1) = -1*vec(:,1);
    assert(det(vec) > 0,'Negative determinant of eigenvector matrix - couldn''t be fixed!');
end

% assemble query point vectors for each eigen-direction
x_qp_vec = [];
dimLengths = [];
for dimIdx = 1:nDim
    sd = sqrt( val(dimIdx,dimIdx));
    x_qp_vec{dimIdx} = -NSD*sd:hss:NSD*sd;
    dimLengths(dimIdx) = length(x_qp_vec{dimIdx});
end
x_qp_raw = cartprod(x_qp_vec)';
x_qp = mu + vec*x_qp_raw;    % now in EIGENSPACE! .. call to cartprod() is fast

% compare to meshgrid for 2D case... need to use transpose!
% meshgrid() and ndgrid() use different conventions!
% https://www.mathworks.com/matlabcentral/answers/99720-what-is-the-difference-between-the-ndgrid-and-meshgrid-functions-in-matlab
% [XX,YY] = meshgrid(x_qp_vec{1},x_qp_vec{2});
% x_qp_diff = x_qp(:,1)-XX(:)

% generate PDF at query points
% [X1Q,X2Q] = meshgrid(xq_vec{1},xq_vec{2});
% XQ = [X1Q(:) X2Q(:)]';
[PQ] = mvnpdf(x_qp',mu_true',cov_true);  % TRUE PDF (vector)
PDF_disp = reshape(PQ,length(x_qp_vec{1}),length(x_qp_vec{2})); % TRUE PDF (array); for display purposes only!
X1Q = reshape(x_qp(1,:),length(x_qp_vec{1}),length(x_qp_vec{2}));
X2Q = reshape(x_qp(2,:),length(x_qp_vec{1}),length(x_qp_vec{2}));

% plot samples and true PDF
figure;
if( ~doUseExternalGibbsFcn & doShowSteps)
    set(gcf,'Position',[1.562000e+02 1.562000e+02 1.194800e+03 6.058000e+02]);
    ax = subplot(2,3,[1 4]);
else
    set(gcf,'Position',[0488 1.562000e+02 0863 6.058000e+02]);
    ax = subplot(1,2,1);
end
title('\bfTrue Density with Samples');
hold on; grid on;
axis equal;
surf(X1Q,X2Q,PDF_disp,'EdgeColor','none','FaceAlpha',0.7);
plot3(x_samp_pre(1,:),x_samp_pre(2,:),ones(size(x_samp_pre,2)),'.','MarkerSize',4,'Color',[ 0 0 0 ]);
plot3(mu(1)+sqrt(val(1,1))*[-vec(1,1) vec(1,1)],mu(2)+sqrt(val(1,1))*[-vec(2,1) vec(2,1)],[2 2],'-','LineWidth',3,'Color',[1 0 1]);
plot3(mu(1)+sqrt(val(2,2))*[-vec(1,2) vec(1,2)],mu(2)+sqrt(val(2,2))*[-vec(2,2) vec(2,2)],[2 2],'-','LineWidth',3,'Color',[1 0 1]);

cmap = colormap;
set(gca,'Color',cmap(1,:));
set(gca,'GridAlpha',0.6);

% plot(xq_cp(1,:),xq_cp(2,:),'r.','MarkerSize',5);  % query point mesh

% apply kernel density smoother to samples
q_samp = ones(1,size(x_samp_pre,2))/size(x_samp_pre,2);

d = size(cov,1);
bw_opt = zeros(1,d);
for i = 1:d
    bw_opt(i) = sqrt(cov(i,i))*(4/((d+2)*size(x_samp_pre,2)))^(1/(d+4));
end
% ks_pdf = part2pdf( x_samp_pre, q_samp, xq_cp, 2.0);  % TODO: this is pretty slow! and appears to generate diagonal-skewed densities?
ks_pdf = mvksdensity(x_samp_pre',x_qp','Kernel','epanechnikov','weights',q_samp,'bandwidth',bwScale*bw_opt);

% reshape PDF
KS_pdf = reshape(ks_pdf,size(X1Q));
if( ~doUseExternalGibbsFcn & doShowSteps)
    ax(end+1) = subplot(2,3,[2 5]);
else
    ax(end+1) = subplot(1,2,2);
end
title('\bfSmoothed Density with Samples & Gibbs');
linkaxes(ax,'xy');
hold on; grid on;
axis equal;
% plot(x_samp(1,:),x_samp(2,:),'.','MarkerSize',5,'Color',0.5*ones(1,3));
surf(X1Q,X2Q,KS_pdf,'EdgeColor','none','FaceAlpha',0.7);
% plot(xq_cp(1,:),xq_cp(2,:),'r.','MarkerSize',5);  %    query point mesh
cmap = colormap;
set(gca,'Color',cmap(1,:));
set(gca,'GridAlpha',0.6);

if(doUseExternalGibbsFcn)
    x_samp_post = gibbsSampleRect(x_qp, ks_pdf, dimLengths, Np, gibbsBurnIn, gibbsM);
else
    % start sampler at random query point
    x0Idx = randi(size(x_qp,2));
    x0 = x_qp(:,x0Idx);
    % plot(x1(1),x0(2),'.','Color',[0 0.8 0],'MarkerSize',20);
    
    % compute number of steps to take
    if(doOneStepOnly)
        Nsteps = 1;
    else
        Nsteps = gibbsBurnIn + 1 + gibbsM*(Np-1);
    end
    
    % start history
    x = x0;
    xIdx = x0Idx;
    x_hist = NaN(length(mu_true),Nsteps+1);
    x_hist(:,1) = x0;
    xIdx_hist = NaN(1,Nsteps+1);
    xIdx_hist(1) = x0Idx;
    
    % show starting point
    if(doShowSteps)
        plot3(x0(1),x0(2),2,'o','MarkerSize',5,'LineWidth',2,'Color',[0 0.8 0]);
        zlim([0 3]);
    end
    
    % iterate sampler
    for gibbsIter = 1:Nsteps
        for dimIdx = 1:nDim
            
            % extract indices for all points in a line along the selected
            % dimension that includes the current point
            subscripts = ndind2sub(dimLengths,xIdx);
            pointSubscripts = repmat(subscripts,length(x_qp_vec{dimIdx}),1);
            pointSubscripts(:,dimIdx) = (1:length(x_qp_vec{dimIdx}))';
            ind = ndsub2ind(dimLengths,pointSubscripts);
            
            % compute 1D PDF and CDF along this line
            pointsAlongDim = x_qp(:,ind);
            distAlongDim = pointsAlongDim - pointsAlongDim(:,1);
            distAlongDim = vecnorm(distAlongDim,2,1);
            normFactor = trapz(distAlongDim,ks_pdf(ind));
            pdf = ks_pdf(ind)/normFactor;
            cdf = cumtrapz(distAlongDim,pdf);
            
            % sample a point from the CDF
            localIdx = find(cdf >= rand(1),1,'first');
            xIdx = ind(localIdx);
            x = x_qp(:,xIdx);
            
            % plot results of stepping along this dimension if requested
            if(doShowSteps)
                subplot(2,3,[2 5]);
                plot3(pointsAlongDim(1,:),pointsAlongDim(2,:),2*ones(size(pointsAlongDim,2)),'k.','MarkerSize',5);
                plot3(x_qp(1,xIdx),x_qp(2,xIdx),2,'ro','MarkerSize',5,'LineWidth',2);
                zlim([0 3]);
                
                % sample many points just to observe distribution
                multiSampleLocalIdx = arrayfun(@(x) find(cdf >= x,1,'first'),rand(1,100));
                
                % compute true PDF along this line
                trueNormFactor = trapz(distAlongDim,PQ(ind));
                truePDF = PQ(ind)/trueNormFactor;
                
                % show 1-D plot
                switch(dimIdx)
                    case 1
                        subplot(2,3,3);
                    case 2
                        subplot(2,3,6);
                    otherwise
                        error('Can only show steps for 2-dim');
                end
                hold on; grid on;
                plot(distAlongDim,pdf,'-','LineWidth',1.6,'Color',[0 0 0.8]);
                plot(distAlongDim,truePDF,'--','LineWidth',1.6,'Color',[0 0 0.8]);
                plot(distAlongDim(multiSampleLocalIdx),0*ones(size(multiSampleLocalIdx)),'r.','MarkerSize',10);
                plot(distAlongDim(localIdx),0,'r+','MarkerSize',10,'LineWidth',3);
                legh = legend('Emperical Density','True Density','Samples','Chosen Sample');
                legh.Position = [0.8607    0.7908    0.1237    0.1148];
                title(sprintf('True Norm Factor: %0.4f',trueNormFactor));
            end
        end
        
        % after stepping once along each eigendirection,
        % store new point as next node in markov chain
        x_hist(:,gibbsIter+1) = x;
        xIdx_hist(gibbsIter+1) = xIdx;
    end
    
    % downsample to keep only selected points from the markov chain
    x_samp_post = x_hist(:,((gibbsBurnIn+1):gibbsM:Nsteps));
end

% show sampled point cloud
if( ~doUseExternalGibbsFcn & doShowSteps)
    subplot(2,3,[2 5]);
else
    subplot(1,2,2);
end
hold on; grid on;
plot3(x_samp_post(1,:),x_samp_post(2,:),ones(size(x_samp_post,2)),'.','MarkerSize',4,'Color',[ 0 0 0 ]);

% principal axes of regularized particle set
mu = mean(x_samp_post,2);
cov = 1/(size(x_samp_post,2)-1)*(x_samp_post-mu)*(x_samp_post-mu)';
[vec,val] = eig(cov);
acos(vec(1,1))*180/pi
% make sure that eigenvector matrix gives a right-handed coordinate system
% (in n-dimensions)... need det(vec) = 1.0 not -1.0
% TODO: may need to further test and refine this!
if(det(vec) < 0)
    vec(:,1) = -1*vec(:,1);
    assert(det(vec) > 0,'Negative determinant of eigenvector matrix - couldn''t be fixed!');
end

% plot principal axes of regularized particle set
plot3(mu(1)+sqrt(val(1,1))*[-vec(1,1) vec(1,1)],mu(2)+sqrt(val(1,1))*[-vec(2,1) vec(2,1)],[2 2],'-','LineWidth',3,'Color',[1 0 1]);
plot3(mu(1)+sqrt(val(2,2))*[-vec(1,2) vec(1,2)],mu(2)+sqrt(val(2,2))*[-vec(2,2) vec(2,2)],[2 2],'-','LineWidth',3,'Color',[1 0 1]);
zlim([0 3]);

%% animate if requested
if(doAnimate & ~doUseExternalGibbsFcn)
    maxStep = min([stepsToAnimate, size(x_hist,2)]);
    figure;
    set(gcf,'Position',[0488 9.220000e+01 5.834000e+02 6.698000e+02]);
    hold on; grid on;
    % plot(x_samp(1,:),x_samp(2,:),'.','MarkerSize',5,'Color',0.5*ones(1,3));
    contour(X1Q,X2Q,KS_pdf,'LineWidth',1.2);
    %     surf(X1Q,X2Q,KS_pdf,'EdgeColor','none','FaceAlpha',0.7);
    % plot(xq_cp(1,:),xq_cp(2,:),'r.','MarkerSize',5);  %    query point mesh
    %     cmap = colormap;
    %     set(gca,'Color',cmap(1,:));
    %     set(gca,'GridAlpha',0.6);
    
    plot3(x0(1),x0(2),2,'.','MarkerSize',30,'LineWidth',2,'Color',[0 0.8 0]);
    xlim([min(x_qp(1,:)) max(x_qp(1,:))]);
    axis equal;
    xlim([min(x_qp(1,:)) max(x_qp(1,:))]);
%     axis off;
    vidAx = gca;
    vidAx.XTickLabel = [];
    vidAx.YTickLabel = [];
    vidAx.XColor = 'none'
    vidAx.YColor = 'none'
    drawnow;
    
    if(doMakeVideo)
        thisImgFile = sprintf('frame%03d.png',gibbsIter);
        saveas(gcf,thisImgFile);
        system(['convert -trim ' thisImgFile ' ' thisImgFile]);  % REQUIRES convert FROM IMAGEMAGICK!
    else
        pause(0.1);
    end
    
    for aniStep = 1:maxStep-1
        plot3(x_hist(1,aniStep+1),x_hist(2,aniStep+1),2,'k.','MarkerSize',5);
        plot3(x_hist(1,aniStep+1),x_hist(2,aniStep+1),2,'ko','MarkerSize',5 ,'LineWidth',2.0);
        plot3(x_hist(1,aniStep+[0 1]),x_hist(2,aniStep+[0 1]),[2 2],'k-','MarkerSize',10);
        
        
        xlim([min(x_qp(1,:)) max(x_qp(1,:))]);
        drawnow;
        
        
        if(doMakeVideo)
            thisImgFile = sprintf('frame%03d.png',aniStep);
            saveas(gcf,thisImgFile);
            system(['convert -trim ' thisImgFile ' ' thisImgFile]);  % REQUIRES convert FROM IMAGEMAGICK!
        else
            pause(0.1);
        end
        
        
    end
    if(doMakeVideo)
        system('ffmpeg -y -r 2 -start_number 1 -i frame%003d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 25 -r 25 output.mp4');     
        system('rm frame*.png');
    end
end

