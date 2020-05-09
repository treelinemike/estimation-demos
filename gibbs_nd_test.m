% Gibbs Sampler: Test in n-dimensions (two, but extensible)
% First use Epaneschnikov kernel denisty smoothing to reconstruct PDF from samples

% restart
close all; clear; clc;

% reset random number generator
rng('default');

% options
Nsteps = 1;
doShowSteps = 1;
doAnimate = 0;
doMakeVideo = 0;
Np = 2000;         % number of particles to sample from the true PDF
Kss = 0.125;       % expand patch of state space to explore by this factor times the span of samples on either side
hss = 1;         % step size in state space TODO: allow different step sizes in each dimension

% true PDF parameters
mu_true = [2 5]';
sigma_1 = 2;
sigma_2 = 4;
corr = 0.8;
cov_true = [sigma_1^2 corr*sigma_1*sigma_2; corr*sigma_1*sigma_2 sigma_2^2];

% sample from the true PDF
x_samp = mvnrnd(mu_true',cov_true,Np)';

% state space to explore (i.e. query points)
NSD = 4;  % number of standard deviations to extend (+/-) in each dimension from mean
nDim = size(x_samp,1);
mu = mean(x_samp,2);
cov = 1/(size(x_samp,2)-1)*(x_samp-mu)*(x_samp-mu)';
[vec,val] = eig(cov);
% choose sign of eigenvectors s.t. rotated basis is within +/- 90deg of
% standard basis
% TODO: may need to further test and refine this!
vec = vec*diag(sign(diag(vec)));
if(det(vec) < 0)
    error('Negative determinant of eigenvector matrix!');
end

% assemble query point vectors for each eigen-direction
xq_vec = [];
dimLengths = [];
for dimIdx = 1:nDim
    sd = sqrt( val(dimIdx,dimIdx));
    xq_vec{dimIdx} = -NSD*sd:hss:NSD*sd;
    dimLengths(dimIdx) = length(xq_vec{dimIdx});
end

% xq_vec = {};
% for dimIdx = 1:size(x_samp,1)
%     dimSpan = max(x_samp(dimIdx,:))-min(x_samp(dimIdx,:));
%     xq_vec{dimIdx} = (min(x_samp(dimIdx,:))-(Kss*dimSpan)):hss:(max(x_samp(dimIdx,:))+(Kss*dimSpan));
% end
xq_cp_raw = cartprod(xq_vec)';
xq_cp = mu + vec*xq_cp_raw; % now in EIGENSPACE! .. call to cartprod() is fast

% compare to meshgrid for 2D case... need to use transpose!
% meshgrid() and ndgrid() use different conventions!
% https://www.mathworks.com/matlabcentral/answers/99720-what-is-the-difference-between-the-ndgrid-and-meshgrid-functions-in-matlab
% [XX,YY] = meshgrid(xq_vec{1},xq_vec{2});
% xq_diff = xq_cp(:,1)-XX(:)


% generate PDF at query points
% [X1Q,X2Q] = meshgrid(xq_vec{1},xq_vec{2});
% XQ = [X1Q(:) X2Q(:)]';
[PQ] = mvnpdf(xq_cp',mu_true',cov_true);  % TRUE PDF (vector)
PDF_disp = reshape(PQ,length(xq_vec{1}),length(xq_vec{2})); % TRUE PDF (array); for display purposes only!
X1Q = reshape(xq_cp(1,:),length(xq_vec{1}),length(xq_vec{2})); 
X2Q = reshape(xq_cp(2,:),length(xq_vec{1}),length(xq_vec{2})); 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            % plot samples and true PDF
figure(1);
% set(gcf,'Position',[0821 0195 0419 0783]);
set(gcf,'Position',[0488 1.562000e+02 0863 6.058000e+02]);
ax = subplot(1,2,1);
title('\bfTrue Density with Samples');
hold on; grid on;
axis equal;
plot(x_samp(1,:),x_samp(2,:),'.','MarkerSize',5,'Color',0.5*ones(1,3));
contour(X1Q,X2Q,PDF_disp,'LineWidth',1.6);
plot(xq_cp(1,:),xq_cp(2,:),'r.','MarkerSize',5);
plot(mu(1)+sqrt(val(1,1))*[-vec(1,1) vec(1,1)],mu(2)+sqrt(val(1,1))*[-vec(2,1) vec(2,1)],'-','LineWidth',2,'Color',[1 0 1]);
plot(mu(1)+sqrt(val(2,2))*[-vec(1,2) vec(1,2)],mu(2)+sqrt(val(2,2))*[-vec(2,2) vec(2,2)],'-','LineWidth',2,'Color',[1 0 1]);

% apply kernel density smoother to samples
q_samp = ones(1,size(x_samp,2))/size(x_samp,2);
ks_pdf = part2pdf( x_samp, q_samp, xq_cp, 2.0);

% reshape PDF
KS_pdf = reshape(ks_pdf,size(X1Q));
ax(end+1) = subplot(1,2,2);
title('\bfSmoothed Density with Samples & Gibbs');
linkaxes(ax,'xy');
hold on; grid on;
axis equal;
plot(x_samp(1,:),x_samp(2,:),'.','MarkerSize',5,'Color',0.5*ones(1,3));
contour(X1Q,X2Q,KS_pdf,'LineWidth',1.6);

% start sampler at random query point
randi(10);
randi(3);
randi(3);
randi(3);
x0Idx = randi(size(xq_cp,2));
x0 = xq_cp(:,x0Idx);
plot(x0(1),x0(2),'.','Color',[0 0.8 0],'MarkerSize',20);

%%
% xq_vec{1} = 1:3;
% xq_vec{2} = 1:4;
% xq_vec{3} = 1:3;
% x0Idx = 6
subscripts = ndind2sub(dimLengths,x0Idx)
% [s1,s2,s3] = ind2sub(dimLengths,x0Idx)



% extract points along dimension 1
dimToExtract = 2;
pointSubscripts = repmat(subscripts,length(xq_vec{dimToExtract}),1);
pointSubscripts(:,dimToExtract) = (1:length(xq_vec{dimToExtract}))';

ind = ndsub2ind(dimLengths,pointSubscripts)
% ind2 = sub2ind(dimLengths,pointSubscripts(:,1),pointSubscripts(:,2),pointSubscripts(:,3))

pointsAlongDim = xq_cp(:,ind);
distAlongDim = pointsAlongDim - pointsAlongDim(:,1);
distAlongDim = vecnorm(distAlongDim,2,1);

subplot(1,2,1);
plot(pointsAlongDim(1,:),pointsAlongDim(2,:),'k.','MarkerSize',5);


figure;
hold on; grid on;
normFactor = trapz(distAlongDim,ks_pdf(ind));
pdf = ks_pdf(ind)/normFactor;
cdf = cumtrapz(distAlongDim,pdf);
plot(distAlongDim,pdf,'b');
plot(distAlongDim,cdf,'r');

figure;




%%



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
        gibbsXQ = repmat(x,1,length(xq_vec{dimIdx}));
        gibbsXQ(dimIdx,:) = xq_vec{dimIdx};
        
        % slice PDF along this dimension at the current point
        switch(dimIdx)
            case 1
                x2idx = find( xq_vec{2} >= x(2),1,'first');
                gibbsPQ = KS_pdf(x2idx,:);
            case 2
                x1idx = find( xq_vec{1} >= x(1),1,'first');
                gibbsPQ = KS_pdf(:,x1idx);
        end
        gibbsPQ = gibbsPQ/trapz(gibbsXQ(dimIdx,:),gibbsPQ);
        gibbsPQ_true = mvnpdf(gibbsXQ',mu_true',cov_true)';
        gibbsPQ_true = gibbsPQ_true/trapz(gibbsXQ(dimIdx,:),gibbsPQ_true);
        
        % compute CDF and sample from it using inverse transform sampling
        gibbsCDF = cumtrapz(gibbsXQ(dimIdx,:),gibbsPQ);
        gibbsPt = gibbsXQ( dimIdx, arrayfun( @(x) find(x <= gibbsCDF,1,'first') , rand(1,1)) );
        x(dimIdx) = gibbsPt;
        gibbsSamp = gibbsXQ( dimIdx, arrayfun( @(x) find(x <= gibbsCDF,1,'first') , rand(1,100)) );  % for illustration only
        
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
            plot(gibbsXQ(dimIdx,:),gibbsPQ,'-','LineWidth',1.6,'Color',[0 0 0.8]);
            plot(gibbsXQ(dimIdx,:),gibbsPQ_true,'--','LineWidth',1.6,'Color',[0 0 0.8]);
            plot(gibbsSamp,0*ones(size(gibbsSamp)),'r.','MarkerSize',10);
            plot(gibbsPt,0,'r+','MarkerSize',10,'LineWidth',3);
            legend('Emperical Density','True Density','Samples','Chosen Sample');
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
        
        if(doMakeVideo)
            thisImgFile = sprintf('frame%03d.png',gibbsIter);
            saveas(gcf,thisImgFile);
            system(['convert -trim ' thisImgFile ' ' thisImgFile]);  % REQUIRES convert FROM IMAGEMAGICK!
        else
            pause(0.1);
        end
    end
    
end

if(doMakeVideo)
    system('ffmpeg -y -r 2 -start_number 1 -i frame%003d.png -vf scale="trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -profile:v high -pix_fmt yuv420p -g 25 -r 25 output.mp4');
    %             system('del frame*.png');
end

% show all points in the markov chain
figure(1);
subplot(1,2,2);
plot(x_hist(1,:),x_hist(2,:),'k.-','MarkerSize',10);
plot(x_hist(1,:),x_hist(2,:),'k.','MarkerSize',10);
plot(x_hist(1,:),x_hist(2,:),'ko','MarkerSize',10,'LineWidth',2.0);

% from Guillaume on Matlab Centeral Answers:
% https://www.mathworks.com/matlabcentral/answers/332718-can-i-store-multiple-outputs-of-a-function-into-a-cell
% note: output is serialized, but not entirely consistent with serialzied
% output from [XX,YY] = meshgrid(xvec,yvec)... need to transpose via XXT = XX'; XXT(:)
% to get same results...
% meshgrid() and ndgrid() use different conventions:
% https://www.mathworks.com/matlabcentral/answers/99720-what-is-the-difference-between-the-ndgrid-and-meshgrid-functions-in-matlab
function p = cartprod(c)
   %returns the cartesian products of the vectors contained in cell array v
    p = cell(size(c));
    [p{:}] = ndgrid(c{:});
    p = cell2mat(cellfun(@(x) x(:), p, 'UniformOutput', false));
end

% index to subscripts in n-dimensions
% without needing to specifcy dimensionality explicitly for output args
function sub = ndind2sub(dimLengths,ind)
dimSizes = [1 cumprod(dimLengths(1:end-1))];
sub = zeros(1,length(dimLengths));
for dimIdx = length(dimLengths):-1:1
    thisIdx = ceil( ind / dimSizes(dimIdx) );
    sub(dimIdx) = thisIdx;
    ind = ind - (thisIdx-1)*dimSizes(dimIdx);
end
end

% subscripts to index in n-dimensions
% without needing to specify dimensionality via separate arguments
% sub = matrix with #col = #dim; #row = #points to evaluate
% ind = vector length = (#row of sub) with indices of each query point 
function ind = ndsub2ind(dimLengths,sub)
dimSizes = [1 cumprod(dimLengths(1:end-1))];
ind = zeros(1,length(dimLengths));

for pointNum = 1:size(sub,1)
   pointIdx = 1;
   for dimIdx = 1:length(dimLengths)
      pointIdx = pointIdx + dimSizes(dimIdx)*(sub(pointNum,dimIdx)-1); 
   end
   ind(pointNum) = pointIdx;
end
end

