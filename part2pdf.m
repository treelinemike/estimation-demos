% follow Simon pg. 473-474 to use the Epanechnikov kernel to smooth our particle estimate
% inputs:
%   x_part: n x m matrix; n = # states; m = # particles
%   q_part: weight for each particle, should sum to 1.0. Could use a flat prior: qi = (1/Np)*ones(1,Np)
%   x_test: points in state space to evaluate PDF (each column is a different point, number of rows = dimension of state space)
% output: 
%   ks_pdf: estimated value of the PDF for each point in x_test
function ks_pdf = part2pdf( x_part, q_part, x_test, bwScaleFactor)

    Ns = size(x_part,1);         % dimension of state space
    Np = size(x_part,2);         % number of particles
    mu = (1/Np)*sum(x_part,2);   % alternativly: mean(x_part,2)
    
    % center the particles and compute covariance matrix
    x_part_ctr = x_part - mu;
    S  = (1/(Np-1))*(x_part_ctr)*(x_part_ctr)';
    
    % square root factorization via Cholesky
    A = chol(S);  % find A s.t. S = A*A'
    
    % compute volume on n-dim sphere (n = dim of state spaces)
    % Simon recommends iterative method
    v = zeros( Ns, 1 );
    v(1) = 2;
    v(2) = pi;
    for n = 3:length(v)
       v(n) = 2*pi*v(n-2)/n;
    end
    vn = v(Ns);
    
    % compute optimal bandwidth
    % Note: eliminatng the 0.5 factor from Simon (i.e. setting bwScaleFactor = 2) will closely approximate
    % MATLAB result; however, Simon argues taht the 0.5 
    h = bwScaleFactor*0.5 * ((8*(1/vn)*(Ns+4)*(2*sqrt(pi))^Ns )^(1/(Ns+4)))* ((Np)^(-1/(Ns+4)));
%     fprintf('Simon BW: %8.6f\n',h);
    
    % initialize PDF
    ks_pdf = zeros(1,size(x_test,2));  % one scalar for each point in state space at which we approximate the PDF
    
    % add contribution from each particle
    for partIdx = 1:Np
        ks_pdf = ks_pdf + q_part(partIdx) * Kh(x_test - x_part(:,partIdx), A, h, vn);
    end
    
end

function result = Kh( x, A, h, vn)
    Ns = size(x,1);
    result = inv(det(A)) * (h^(-1*Ns))*K( (A\x)/h , vn) ;
 
end

% Epanechnikov kernel
function result = K( x, vn )
    Ns = size(x,1);
    normx = vecnorm(x,2,1);
    result = ((Ns+2)/(2*vn))*(1-(normx.^2));
    result( normx >= 1) = 0;
end