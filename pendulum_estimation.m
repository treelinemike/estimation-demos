% restart
close all; clear all; clc;

% general options
doAnimateSystem = 0;


% simulation time parameters
COV_w_true = [0.01^2 0; 0 0.0875^2];   % covariance matrix for state propagation noise (note: rows correspond to errors in DERIVATIVES of state variables)
t0 = 0;       % [s] simulation start time
tf = 20;      % [s] simulation end time
dt = 0.01;     % [s] timestep size
opts = odeset('RelTol',1e-8,'AbsTol',1e-12);

% sampling options
COV_v = (0.002)^2;    % VARIANCE of sensor noise; assumed to be well known (truth value = value used in estimator)

% estimator options
Np = 2000;                       % number of particles
COV_w = [0.04^2 0; 0 0.2^2];   % assumed covariance matrix for state propagation noise (note: rows correspond to errors in DERIVATIVES of state variables)

% define parameters of physical system in a structure
% that we can pass through the ODE solver to the update function
sysParams = [];
sysParams.m = 2;   
sysParams.l = 1;
sysParams.c = 1;
sysParams.g = 9.81;
sysParams.w_t = [0 0]';

% parameters for deterministic system
sysParamsD = sysParams;
sysParamsD.w_t = [0 0]'; % just to be explicit

% parameters for deterministic, undamped system
sysParamsDUD = sysParamsD;
sysParamsDUD.c = 0;

% parameters for undamped system --- THIS IS THE MODEL ASSUMED IN THE
% ESTIMATOR
sysParamsUD = sysParams;
sysParamsUD.c = 0;

% initial conditions (state vector: [theta theta_dot]' stacked as: [stochastic truth; undamped model propagation; deterministic truth])
theta_t_0     = 25*pi/180;      % [rad]
theta_t_dot_0 = 0;              % [rad/s]
theta_m_0     = 25*pi/180;      % [rad]
theta_m_dot_0 = 0;              % [rad/s]
X0 = [theta_t_0 theta_t_dot_0 theta_m_0 theta_m_dot_0 theta_t_0 theta_t_dot_0]';  % [rad rad/s rad rad/s rad rad/s]'
X = X0;

% data storage
time = [t0];
data = [X0];
theta_ddot =  -(sysParams.c/(sysParams.m*sysParams.l^2))*X0(2)-1*(sysParams.g/sysParams.l)*sin(X0(1))*ones(2,1);

% reset random number generator using consistant seed
% although noise is random it will be the same for every run
rng(2374,'twister');

% run simulation
for t = t0:dt:(tf-dt)
    
    % calculate timestep for ODE solving
    odeTime = [t t+dt];
    
    % generate the random noise for this timestep
    % generate state propagation noise
    % want to keep the random draw CONSTANT inside of the ODE45 update
    % function, so draw it here and pass it in
    sysParams.w_t = mvnrnd([0 0]',COV_w_true,1)';

    % propagate state
    [T,X1] = ode45(@(t,X) propDynamics(t,X,sysParams),odeTime,X(1:2,1),opts);
    [T,X2] = ode45(@(t,X) propDynamics(t,X,sysParamsDUD),odeTime,X(3:4,1),opts);
    [T,X3] = ode45(@(t,X) propDynamics(t,X,sysParamsD),odeTime,X(5:6,1),opts);
    
    
    X1 = X1(end, :)';  % note: this step is necessary to keep state vector dimensions correct for next call to ode45()
    X2 = X2(end, :)';
    X3 = X3(end, :)';
    X = [X1;X2;X3];
    
    % store results from this timestep
    time(end+1)   = T(end);
    data(:,end+1) = X; % note: discarding state values at intermediate timesteps calculated by ode45()
    theta_ddot(:,end+1) = -(sysParams.c/(sysParams.m*sysParams.l^2)).*[X(2);X(6)]-1*(sysParams.g/sysParams.l).*sin([X(1);X(5)]);
end

%%
omega_n = sqrt(sysParams.g/sysParams.l)
tau_n = 2*pi/omega_n
c_cr_eq = 2*omega_n;
zeta = (sysParams.c/(sysParams.m*sysParams.l^2))/c_cr_eq;
omega_d = omega_n*sqrt(1-zeta^2)
tau_d = 2*pi/omega_d
%% plot time series for detail trajectory
ph = [];
figure;
set(gcf,'Position',[0345 0188 1275 0505]);
ax = subplot(2,1,1);
hold on; grid on;
plot(time,0*ones(1,length(time)),'k--');
ph(end+1) = plot(time,data(1,:),'Color',[0 0.7 0],'LineWidth',1.6);
ph(end+1) = plot(time,data(3,:),'Color',[0.7 0 0],'LineWidth',1.6);
xlabel('\bfTime [s]');
ylabel('\bf Angular Position [rad]');
legend(ph,'Truth','Model');
xlim([min(time) max(time)]);

% now compute measurements of vertical displacement of pendulum tip from
% center of rotation (observed quantity)

ax(end+1) = subplot(2,1,2);
hold on; grid on;
z_true = sysParams.l*cos(data(1,:));
linkaxes(ax,'x');

% sample the true signal
dt_samp = 0.25;  % observation sampling period
t_samp = dt_samp:dt_samp:time(end);  % NOTE: DO NOT sample at the initial state (we assume that the initial state estimate is given/known)
z_samp = interp1(time,z_true,t_samp)' + sqrt(COV_v)*randn(length(t_samp),1);
x_samp = interp1(time,data(1:2,:)',t_samp)';
plot(time,z_true,'Color',[0 0.7 0],'LineWidth',1.6);
plot(t_samp,z_samp,'.','MarkerSize',20,'Color','m');

%% show angular position, velocity, and acceleration for the pendulum
% these are the TRUTH traces
% one is the deterministic solution
% the other is the stochastic solution which we will take as acutal truth
% for the estimator
figure;
set(gcf,'Position',[0697 0122 0550 0822]);
ax2 = subplot(3,1,1);
hold on; grid on;

plot(time,data(5,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot(time,data(1,:),'b:','LineWidth',1.6);
legend('Stochastic Truth','Deterministic Evolution')
title('\bfAngular Position');
xlabel('\bfTime [s]');
ylabel('\bf[rad]');

ax2(end+1) = subplot(3,1,2);
hold on; grid on;
plot(time,data(6,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot(time,data(2,:),'b:','LineWidth',1.6);
title('\bfAngular Velocity');
xlabel('\bfTime [s]');
ylabel('\bf[rad/s]');

ax2(end+1) = subplot(3,1,3);
hold on; grid on;
plot(time,theta_ddot(2,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot(time,theta_ddot(1,:),'b:','LineWidth',1.6);
% plot(time,gradient(data(6,:),time),'r--','LineWidth',1.6);  % to check computation of acceleration
title('\bfAcceleration');
xlabel('\bfTime [s]');
ylabel('\bf[rad/s^2]');

linkaxes(ax2,'x');
xlim([0,max(time)]);

%% Animate result in a new plot
if(doAnimateSystem)
    figure;
    hold on; grid on;
    
    % animate each frame of results
    for tIdx = 1:size(data,2)
        
        % extract state at current timestep
        theta = data(1,tIdx);
        theta_dot = data(2,tIdx);
        
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
    end
end

%%
% now we have a true trajectory and a set of observations
% we will step through time and filter to the current timestep

% first, draw a sample of particles from the initial state
mu = data(1:2,1);
COV = COV_w
Xp = mvnrnd(mu',COV,Np)';
x_true = data(:,1);




% step through time
% evaluating one "frame" at a time moving from
% local state 1 (beginning of frame) to local state 2 (end of frame)
% note that the first observation is NOT at the initial time b/c we assume
% that we have an initial state estimate
for obsIdx = 1:1%length(t_samp)
    
    % initialize figure
    figure;
    set(gcf,'Position',[0207 0346 1527 0420]);
    
    % show particles at start of the estimation frame (state 1)
    subplot(1,3,1);
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
        x_prev = Xp(:,particleIdx);
        
        % compute "proposal distribution" by propagating particles forward
        % in time through system model
        odeTime = [0 dt_samp];
        sysParamsUD.w_t = mvnrnd([0 0]',COV_w,1)';
        [T,X] = ode45(@(t,X) propDynamics(t,X,sysParamsUD),odeTime,x_prev,opts);
        x_prior(:,particleIdx) = X(end,:)';
        
        % get observation and compute innovation/residual "r"
        r(particleIdx) = z_samp(obsIdx) - sysParamsUD.l*cos( x_prior(1,particleIdx) );
        
        % compute particle weight, using a simple gaussian centered at zero
        % innovation
        % could use a different weighting scheme here
        %%%%% TODO: DO THIS OUTSIDE OF PARTICLE LOOP! %%%%%%%%%
        w(particleIdx) = expm(-0.5*r(particleIdx)'*inv(COV_v)*r(particleIdx)); % using expm even though the argument happens to be a scalar in this case
        
    end
    
    % normalize weights
    w = w ./sum(w);
    wCDF = cumsum(w);
    
    % sample from CDF (uses universality of the uniform distribution to
    % generate samples)
    % and assemble posterior
    resampIdx = arrayfun(@(afin1) find( afin1 <= wCDF,1,'first'), rand(Np,1));
    x_post = x_prior(:,unique(resampIdx));  % note: unique discards duplicates, thinning the particle ensemble
    mu = mean(x_post,2);
    COV = (1/(Np-1))*(x_post-mu)*(x_post-mu)';
    
    % plot proposal/prior distribution
    subplot(1,3,1);
    plot(x_prior(1,:),x_prior(2,:),'.','MarkerSize',5,'Color',[1 .6 .6]);
    plot(x_true(1),x_true(2),'o','MarkerSize',20,'Color',[0 0.5 0],'LineWidth',3);
    plot(x_samp(1,:),x_samp(2,:),'.','Color',[0 0.5 0],'MarkerSize',10);
    mu = mean(x_prior,2);
    plot(mu(1),mu(2),'bo','MarkerSize',10,'LineWidth',3);
    plot(data(1,:),data(2,:),'-','Color',[0 0.5 0],'LineWidth',1); % true trajectory in state space
    plot(data(3,:),data(4,:),'-','Color',[0 0 0.5],'LineWidth',1); % assumed model trajectory in state space (deterministic, no damping)
        
    % plot innovation
    subplot(1,3,2);
    hold on; grid on;
    plot(r,zeros(size(r)),'b.','MarkerSize',5);
    
    % functional form of weighting function
    w_func_domain = -3*sqrt(COV_v):0.0001:3*sqrt(COV_v);
    w_func = exp(-0.5*inv(COV_v)*(w_func_domain.^2));
    plot(w_func_domain,w_func,'-');
    xlabel('\bfInnovation / Residual [m]');
    ylim([0,2]);
    
    % plot posterior distribution
    subplot(1,3,1);
    plot(x_post(1,:),x_post(2,:),'k.','MarkerSize',2);
    mu = mean(x_post,2);
    plot(mu(1),mu(2),'k*','MarkerSize',10,'LineWidth',3);
    
    Xp = x_post; 
    mu = mean(x_post,2);
    Np = size(x_post,2);
end

% function to propagate state for ODE solver
function  Xdot = propDynamics(t,X,sysParams)

% recover paramters
m = sysParams.m;
l = sysParams.l;
c = sysParams.c;
g = sysParams.g; 
w_t = sysParams.w_t;

% deconstruct state vector
theta = X(1);
theta_dot = X(2);

% construct Xdot from differential equation
% note:     X    = [theta      theta_dot]
% therefore Xdot = [theta_dot  theta_ddot] + w_t
Xdot = zeros(2,1);

% stochastic truth
Xdot(1,:) = theta_dot + w_t(1);
Xdot(2,:) = -(c/(m*l^2))*theta_dot-1*(g/l)*sin(theta) + w_t(2);
end
