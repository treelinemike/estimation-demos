% restart
close all; clear all; clc;

% reset random number generator using consistant seed
% although noise is random it will be the same for every run
% rng(2374,'twister');
rng(4265,'twister');    
% rng(111,'twister');   

% general options
doAnimateSystem = 0;
doShowDynamicsPlots = 1;

% simulation time parameters
t0 = 0;        % [s] simulation start time
tf = 20;       % [s] simulation end time
dt = 0.001;    % [s] simulation timestep size (discritized dynamics)

% sampling options
dt_samp = 0.05;      % observation sampling period
COV_v = (0.002)^2;  % VARIANCE of sensor noise; assumed to be well known (truth value = value used in estimator)

% estimator options
Np = 2000;                                  % number of particles
COV_0 = [(5*pi/180)^2 0; 0 (30*pi/180)^2];  % covariance of Gaussian from which we'll select initial paarticles; NOTE: stdev units are rad and rad/s (NOT DERIVATIVES) for this specific covariance matrix
COV_w = [0.04^2 0; 0 0.2^2];                % assumed covariance matrix for state propagation noise (note: rows correspond to errors in DERIVATIVES of state variables)

% initial conditions (state vector: [theta theta_dot]' stacked as: [stochastic truth; undamped model propagation; deterministic truth])
theta_0     = 25*pi/180;      % [rad]
theta_dot_0 = 0;              % [rad/s]
x0 = [theta_0 theta_dot_0]';  % [rad rad/s rad rad/s rad rad/s]'

% define parameters of physical system in a structure
% that we can pass through the ODE solver to the update function
sysParams = [];
sysParams.m = 2;
sysParams.l = 1;
sysParams.c = 1;
sysParams.g = 9.81;
sysParams.COV_w_true = [0.0002^2 0; 0 0.0003^2];

% SD: parameters for Stochastic, Damped system
% this is the "true" signal
sysParamsSD = sysParams;

% DD: parameters for Deterministic, Damped system
% used for comparison to the "true" signal, highlighting stochastic effects
sysParamsDD = sysParams;
sysParamsDD.COV_w_true = zeros(2,2);

% DU: parameters for Deterministic, Undamped system
% this is how a naive model would propagate system trajectory
sysParamsDU = sysParams;
sysParamsDU.c = 0;
sysParamsDU.COV_w_true = zeros(2,2);

% SU: parameters for Stochastic, Undamped system
% THIS IS THE MODEL ASSUMED IN / USED BY THE ESTIMATOR
sysParamsSU = sysParams;
sysParamsSU.c = 0;


% compute undamped and damped frequencies and time constants
omega_n = sqrt(sysParams.g/sysParams.l);
tau_n = 2*pi/omega_n;
c_cr_eq = 2*omega_n;
zeta = (sysParams.c/(sysParams.m*sysParams.l^2))/c_cr_eq;
omega_d = omega_n*sqrt(1-zeta^2);
tau_d = 2*pi/omega_d;

% compute number of steps to take for discrete system
% and assemble time vector
Nsteps = ceil(tf/dt);
time = (0:Nsteps)*dt;

% run discrete simulations (note: nonlinear state transition function called! error ~ O(dt) )
x_SD = stepDynamics(Nsteps,dt,x0,sysParamsSD);
x_DD = stepDynamics(Nsteps,dt,x0,sysParamsDD);
x_DU = stepDynamics(Nsteps,dt,x0,sysParamsDU);
x_SU = stepDynamics(Nsteps,dt,x0,sysParamsSU);

% sample the true signal per measurement model
z_true = sysParams.l*cos(x_SD(1,:));
t_samp = dt_samp:dt_samp:time(end);  % NOTE: DO NOT sample at the initial state (we assume that the initial state estimate is given/known)
z_samp = interp1(time,z_true,t_samp)' + sqrt(COV_v)*randn(length(t_samp),1);
x_samp = interp1(time,x_SD',t_samp)';

%% plot time series trajectories
if(doShowDynamicsPlots)
    ph = [];
    figure;
    set(gcf,'Position',[0345 0188 1275 0505]);
    ax = subplot(2,1,1);
    hold on; grid on;
    plot(time,0*ones(1,length(time)),'k--');
    ph(end+1) = plot(time,x_DU(1,:),'Color',[0.8 0 0],'LineWidth',1.6);
    ph(end+1) = plot(time,x_SD(1,:),'Color',[0 0 0.8],'LineWidth',1.6);
    xlabel('\bfTime [s]');
    ylabel('\bf Angular Position [rad]');
    legend(ph,'Naive Model','Truth','Location','SouthEast');
    xlim([min(time) max(time)]);

    ax(end+1) = subplot(2,1,2);
    hold on; grid on;
    plot(time,z_true,'Color',[0 0 0.8],'LineWidth',1.6);
    plot(t_samp,z_samp,'.','MarkerSize',20,'Color','m');
    ylabel('\bfObservation');
    legend('Truth','Samples','Location','SouthEast');
    linkaxes(ax,'x');
end

%% show angular position, velocity, and acceleration for the pendulum
% these are the TRUTH traces
% one is the deterministic solution
% the other is the stochastic solution which we will take as acutal truth
% for the estimator
if(doShowDynamicsPlots)
    figure;
    set(gcf,'Position',[0697 0122 0550 0822]);
    ax2 = subplot(3,1,1);
    hold on; grid on;
    plot(time,x_DU(1,:),'-','LineWidth',1.6,'Color',[0.8 0 0]);
    plot(time,x_DD(1,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
    plot(time,x_SD(1,:),'-','LineWidth',1.0,'Color',[0 0 0.8]);
    legend('Deterministic Undamped','Deterministic Damped','Stochastic "Truth"');
    title('\bfAngular Position');
    xlabel('\bfTime [s]');
    ylabel('\bf[rad]');
    
    ax2(end+1) = subplot(3,1,2);
    hold on; grid on;
    plot(time,x_DD(2,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
    plot(time,x_SD(2,:),'-','LineWidth',1.0,'Color',[0 0 0.8]);
    title('\bfAngular Velocity');
    xlabel('\bfTime [s]');
    ylabel('\bf[rad/s]');
    
    ax2(end+1) = subplot(3,1,3);
    hold on; grid on;
    plot(time,gradient(x_DD(2,:),time),'LineWidth',1.6,'Color',[0 0.8 0]);
    plot(time,gradient(x_SD(2,:),time),'-','LineWidth',1.0,'Color',[0 0 0.8]);
    % plot(time,gradient(data(6,:),time),'r--','LineWidth',1.6);  % to check computation of acceleration
    title('\bfAcceleration');
    xlabel('\bfTime [s]');
    ylabel('\bf[rad/s^2]');
    
    linkaxes(ax2,'x');
    xlim([0,max(time)]);
end

%% animate "true" pendulum in a new plot
if(doAnimateSystem)
    figure;
    hold on; grid on;
    
    % animate each frame of results
    for tIdx = 1:floor(0.05/dt):size(x_SD,2)
        
        % extract position at current timestep
        theta = x_SD(1,tIdx);
        
        % recover length
        l = sysParams.l;
        
        % determine bob location
        bob_x = l*sin(theta);
        bob_y = -l*cos(theta);
        
        % clear axes and start plotting the current frame
        cla;
        
        % plot XYZ (all black) and xyz (x=red, y=green, z=blue) coordinate frames
        plot(0,0,'k.','MarkerSize',30);
        plot([0 bob_x],[0 bob_y],'k-','LineWidth',6);
        plot(bob_x,bob_y,'o','MarkerSize',30,'LineWidth',6,'MarkerFaceColor',[1 1 1]);
        
        % finish formatting axes
        axis equal;
        xlabel('\bfX');
        ylabel('\bfY');
        xlim([-1.2*l 1.2*l]);
        ylim([-1.2*l 1.2*l]);
        drawnow;
        pause(0.01);
    end
end

%%
% now we have a true trajectory and a set of observations
% we will step through time and filter to the current timestep

% first, draw a sample of particles from the initial state
x_true = x_SD(:,1);
Xp = mvnrnd(x_true',COV_0,Np)';
mu = mean(Xp,2);
COV = (1/(Np-1))*(Xp-mu)*(Xp-mu)';

% step through time
% evaluating one "frame" at a time moving from
% local state 1 (beginning of frame) to local state 2 (end of frame)
% note that the first observation is NOT at the initial time b/c we assume
% that we have an initial state estimate
for obsIdx = 1:1%length(t_samp)
    
    % initialize figure
    figure;
    set(gcf,'Position',[0198 0102 1527 0833]);
    
    % show particles at start of the estimation frame (state 1)
    subplot(2,3,1);
    title('\bfEvolution in State Space');
    hold on; grid on;
    plot(Xp(1,:),Xp(2,:),'.','MarkerSize',5,'Color',[.6 .6 1]);
    plot(mu(1),mu(2),'bo','MarkerSize',10,'LineWidth',3);
    plot(x_true(1),x_true(2),'o','MarkerSize',20,'Color',[0 0.5 0],'LineWidth',3);
    %     plot(data(1,:),data(2,:),'-','Color',[0 0.5 0],'LineWidth',1); % true trajectory in state space
    xlabel('\bfx_1: Angular Position [rad]');
    ylabel('\bfx_2: Angular Velocity [rad/s]');
    
    % get new truth and observation
    x_true = x_samp(:,obsIdx);
    
    % now propigate and update each particle
    x_prior = zeros(2,Np);
    r = zeros(1,Np);
    w = zeros(1,Np);
    for particleIdx = 1:Np
        
        % get previous posterior particle set
        x_prev_i = Xp(:,particleIdx);
        
        % compute "proposal distribution" by propagating particles forward
        % in time through naive DU (deterministic, undamped) system model
        odeTime = [0 dt_samp];
        x_prior_i = stepDynamics(floor(dt_samp/dt),dt,x_prev_i,sysParamsDU);
        
        x_prior(:,particleIdx) = x_prior_i(:,end);
        
        % get observation and compute innovation/residual "r"
        r(particleIdx) = z_samp(obsIdx) - sysParamsSD.l*cos( x_prior(1,particleIdx) );
        
        % compute particle weight, using a simple gaussian centered at zero
        % innovation
        % could use a different weighting scheme here
        %%%%% TODO: DO THIS OUTSIDE OF PARTICLE LOOP! %%%%%%%%%
        w(particleIdx) = expm(-0.5*r(particleIdx)'*inv(COV_v)*r(particleIdx)); % using expm even though the argument happens to be a scalar in this case
        
    end
    
    % compute mean and standard deviation of prior particle set
    x_prior_mean_1 = mean(x_prior(1,:));
    x_prior_ctr_1 = x_prior(1,:)-x_prior_mean_1;
    x_prior_stdev_1 = sqrt( 1/(length(x_prior_ctr_1)-1)*(x_prior_ctr_1*x_prior_ctr_1'));
    assert( abs(x_prior_stdev_1 - std(x_prior(1,:)))/std(x_prior(1,:)) < 0.01 , 'Standard deviation calculation not within 1%!');
    
    % normalize weights to sum to 1.0
    w = w ./sum(w);
    
    % compute "CDF" in column 3
    wCDF_all = [x_prior(1,:)',w',w',(1:Np)'];
    wCDF_all = sortrows(wCDF_all,1,'ascend');
    wCDF_all(:,3) = cumsum(wCDF_all(:,2));
    
    % sample from CDF (uses "inverse transform sampling" / "universality of the uniform" to generate samples)
    % and assemble posterior
    % DO NOT USE UNIQUE TO THIN PARTICLES HERE!!!! THIS RESULTS IN THE
    % WRONG PDF!!! WE WILL HAVE DUPLICATE PARTICLES.
    resampIdx = arrayfun(@(afin1) find( afin1 <= wCDF_all(:,3),1,'first'), rand(Np,1));
    x_post = x_prior(:,wCDF_all(resampIdx,4));

    
    
    mu = mean(x_post,2);
    COV = (1/(Np-1))*(x_post-mu)*(x_post-mu)';
    
    % plot proposal/prior distribution
    subplot(2,3,1);
    plot(x_prior(1,:),x_prior(2,:),'.','MarkerSize',5,'Color',[1 .6 .6]);
    plot(x_true(1),x_true(2),'o','MarkerSize',20,'Color',[0 0.5 0],'LineWidth',3);
    plot(x_samp(1,:),x_samp(2,:),'.','Color',[0 0.5 0],'MarkerSize',10);
    fprintf('Prior: (%8.4f,%8.4f); Truth: (%8.4f,%8.4f); Observation: %8.4d\n',mu(1),mu(2),x_true(1),x_true(2),z_samp(obsIdx));
    plot(mu(1),mu(2),'bo','MarkerSize',10,'LineWidth',3);
    
    % show true trajectory and one period of the undamped phase portrait
    % (only one period b/c Fwd Euler inaccuracy makes oscillations grow)
    plot(x_SD(1,:),x_SD(2,:),'-','Color',[0 0 0.8],'LineWidth',1); % true trajectory in state space
    lowerBoundIdx = find(time >= 0.8*tau_n, 1, 'first');
    upperBoundIdx = find(time >= 1.2*tau_n, 1, 'first');
    [~,onePerIdx] = max( x_DU(1,lowerBoundIdx:upperBoundIdx) );
    onePerIdx = onePerIdx + lowerBoundIdx;
    plot(x_DU(1,1:onePerIdx),x_DU(2,1:onePerIdx),'-','Color',[0.8 0 0],'LineWidth',1); % assumed model trajectory in state space (deterministic, no damping)
    
    % plot innovation
    subplot(2,3,2);
    title('\bf  Observation');
    hold on; grid on;
    plot(r,zeros(size(r)),'b.','MarkerSize',5);
    
    % functional form of weighting function
    w_func_domain = -3*sqrt(COV_v):0.0001:3*sqrt(COV_v);
    w_func = exp(-0.5*inv(COV_v)*(w_func_domain.^2));
    plot(w_func_domain,w_func,'-');
    xlabel('\bfInnovation / Residual [m]');
    ylim([0,1.5 ]);
    
    mu = mean(x_post,2);
    %     x_post = x_prior + ( mean(x_post,2) - mean(x_prior,2)); % use all of the prior particles, just shift them to the "new" centroid based on observation
    Xp = x_post;
    Np = size(x_post,2);
    
    
    % plot posterior distribution after resampling
    subplot(2,3,1);
    plot(x_post(1,:),x_post(2,:),'k.','MarkerSize',2);
    mu = mean(x_post,2);
    plot(mu(1),mu(2),'k*','MarkerSize',10,'LineWidth',3);
    
    % show bayesian update in 1D
    subplot(2,3,4:6);
    x_test = -pi/2:0.0001:pi/2;
    LH = zeros(size(x_test));  % likelihood function
    for xIdx = 1:length(x_test)
        LH(xIdx) = normpdf( z_samp(obsIdx) , sysParams.l*cos( x_test(xIdx) ), sqrt(COV_v)  );
    end
    hold on; grid on;
    %     prior_pdf = normpdf( x_test, x_prior_mean_1, x_prior_stdev_1 );
    prior_pdf_ks = ksdensity(x_prior(1,:),x_test,'Kernel','epanechnikov');
    LH_norm = LH/trapz(x_test,LH);
    post_pdf = prior_pdf_ks .* LH;
    post_pdf = post_pdf / trapz(x_test,post_pdf);
%     plot(x_true(1)*ones(2,1),[-1 max( [max(LH_norm), max(prior_pdf_ks), max(post_pdf)])],':','LineWidth',1.6,'Color',[0 0.8 0]);
    plot(x_test,prior_pdf_ks,'r-','LineWidth',3.6);
    plot(x_test,LH_norm,'b-','LineWidth',3.6);
%     plot(x_test,post_pdf_est,'m-','LineWidth',1.6);
    post_pdf_est = prior_pdf_ks.*LH_norm;
    post_pdf_est = post_pdf_est / trapz(x_test,post_pdf_est);
    plot(x_test,post_pdf_est,'m-','LineWidth',3.6);

    % test particle propigation using samples
    % need to sort so that trapz() works correctly
    ppdf = [x_prior(1,:)' w'];
    ppdf = sortrows(ppdf,1,'ascend');
    ppdf(:,2) = ppdf(:,2)/trapz(ppdf(:,1),ppdf(:,2));
%     plot(ppdf(:,1),ppdf(:,2),'k.','MarkerSize',10);
    
    % estimate density from the resampled PDF
    post_rs_ks = ksdensity(x_post(1,:),x_test,'Kernel','epanechnikov');
%     plot(x_test,post_rs_ks,'--','Color',[0 0.8 0],'LineWidth',1.6);
    xlim(x_true(1)+[-0.1 0.1]);
    
    
    legend('Truth','Prior (Epanechnikov Kernel)','Normalized Likelihood','Posterior','PDF from Particle Weights','Smoothed Posterior Particles','Location','NorthWest');
    xlabel('\bfx_1: Angular Position [rad]');
    ylabel('\bfProbability Density');
    
    
    
    subplot(2,3,3);
    hold on; grid on;
    xlim(x_true(1)+[-0.05 0.05]);
    wpdf = wCDF_all(:,2)/trapz(wCDF_all(:,1),wCDF_all(:,2));
    intCDF = cumtrapz(wCDF_all(:,1),wpdf);
    plot(wCDF_all(:,1),wpdf,'r-');
    plot(wCDF_all(:,1),100*wCDF_all(:,3),'b');   
    plot(wCDF_all(:,1), 100*intCDF,'m-');
    
    plot(x_post(1,:),zeros(size(x_post)),'m.','MarkerSize',10);
    x_test = -pi/2:0.0001:pi/2;
    ppdf = ksdensity(x_post(1,:),x_test,'Kernel','epanechnikov');
    plot(x_test,ppdf,'m-');
    
    % try again to sample from this pdf
    samp2 = zeros(1,Np);
    for i = 1:length(samp2)
        samp2(i) = wCDF_all( find( rand <= wCDF_all(:,3), 1, 'first'),1);
    end
    plot(samp2,zeros(size(samp2)),'c.','MarkerSize',5);
    ppdf2 = ksdensity(samp2,x_test,'Kernel','epanechnikov');
    plot(x_test,ppdf2,'C-');
    
%     error('done');
    
    
end

% function to propagate state via finite differences (discritized dynamics)
function x_traj = stepDynamics(N,dt,x,sysParams)
    x_traj = zeros(size(x,1),N+1);
    x_traj(:,1) = x;
    
    for i = 1:N
        x_next = zeros(size(x)); 
        x_next(1) = x(1)+dt*x(2);
        x_next(2) = (1- (sysParams.c*dt/(sysParams.m*sysParams.l^2)))*x(2) - (sysParams.g*dt/sysParams.l)*sin(x(1));
        x = x_next + mvnrnd([0 0]',sysParams.COV_w_true,1)';
        x_traj(:,i+1) = x;
    end
end