% NOTE: CONSIDER REMOVING LATEX INTERPRETER IN LEGENDS B/C MAC AND SOME
% OLDER VERSIONS OF MATLAB THROW ERRORS WITH THAT

% plot phase portraits for the simple pendulum

% restart
close all; clear all; clc;

% reset random number generator using consistant seed
% although noise is random it will be the same for every run
rng(2374,'twister');

% simulation time parameters
SIGMA_w_true = [0.01^2 0; 0 0.0875^2];   % covariance matrix for state propigation noise (note: rows correspond to errors in DERIVATIVES of state variables)
t0 = 0;       % [s] simulation start time
tf = 20;      % [s] simulation end time
dt = 0.1;     % [s] timestep size
opts = odeset('RelTol',1e-8,'AbsTol',1e-12);

% sampling options
sigma_v = 0.002;    % standard deviation of sensor noise; assumed to be well known (truth value = value used in estimator)

% estimator options
SIGMA_w = [0.002^2 0; 0 0.1^2];   % assumed covariance matrix for state propigation noise (note: rows correspond to errors in DERIVATIVES of state variables)


% define parameters for cases to analyze
sysParams = [];
sysParams.m = 2;   
sysParams.l = 1;
sysParams.c = 1;
sysParams.g = 9.81;
sysParams.SIGMA_w_true = SIGMA_w_true;

% initial conditions (state vector: [theta theta_dot]')
theta_t_0     = 25*pi/180;      % [rad]
theta_t_dot_0 = 0;              % [rad/s]
theta_m_0     = 25*pi/180;      % [rad]
theta_m_dot_0 = 0;              % [rad/s]
X0 = [theta_t_0 theta_t_dot_0 theta_m_0 theta_m_dot_0 theta_t_0 theta_t_dot_0]';  % [rad rad/s rad rad/s]'
X = X0;

% data storage
time = [t0];
data = [X0];
theta_ddot =  -(sysParams.c/(sysParams.m*sysParams.l^2))*X0(2)-1*(sysParams.g/sysParams.l)*sin(X0(1))*ones(2,1);

% run simulation
for t = t0:dt:(tf-dt)
    
    % calculate timestep for ODE solving
    odeTime = [t t+dt];
    
    % generate the random noise for this timestep
    % generate state propigation noise
    % want to keep the random draw CONSTANT inside of the ODE45 update
    % function, so draw it here and pass it in
    sysParams.w_t = mvnrnd([0 0]',SIGMA_w_true,1)';

    % propigate state
    [T,X] = ode45(@(t,X) propDynamics(t,X,sysParams),odeTime,X,opts);
    X = X(end, :)';  % note: this step is necessary to keep state vector dimensions correct for next call to ode45()
    
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
t_samp = 0:dt_samp:time(end);
z_samp = interp1(time,z_true,t_samp)' + sigma_v*randn(length(t_samp),1);
plot(time,z_true,'Color',[0 0.7 0],'LineWidth',1.6);
plot(t_samp,z_samp,'.','MarkerSize',20,'Color','m');

%%
figure;
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
% figure;
% hold on; grid on;
% 
% % animate each frame of results
% for tIdx = 1:size(data,2)
%     
%     % extract state at current timestep
%     theta = data(1,tIdx);
%     theta_dot = data(2,tIdx);
%    
%     % recover length
%     l = sysParams.l;
%     
%     % determine bob location
%     bob_x = l*sin(theta);
%     bob_y = -l*cos(theta);
%     
%     % clear axes and start plotting the current frame
%     cla;
%     
%     % plot XYZ (all black) and xyz (x=red, y=green, z=blue) coordinate frames
%     plot(0,0,'k.','MarkerSize',30);
%     plot([0 bob_x],[0 bob_y],'k-','LineWidth',6);
%     plot(bob_x,bob_y,'o','MarkerSize',30,'LineWidth',6,'MarkerFaceColor',[1 1 1]);
%     
%     % finish formatting axes
%     axis equal;
%     xlabel('\bfX');
%     ylabel('\bfY');
%     xlim([-1.2*l 1.2*l]);
%     ylim([-1.2*l 1.2*l]);
% 	drawnow;
% end

% now we have a true trajectory and a set of observations
% we will step through time and filter to the current timestep

% first, draw a sample of particles from the initial state
% Sigma1 = diag(
% Xp1 = X0

for tidx = 1:length(time)
    
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
theta_t = X(1);
theta_t_dot = X(2);
theta_m = X(3);
theta_m_dot = X(4);
theta_t_det = X(5);
theta_t_dot_det = X(6);

% construct Xdot from differential equation
% note:     X    = [theta_t      theta_t_dot   theta_m      theta_m_dot]
% therefore Xdot = [theta_t_dot  theta_t_ddot  theta_m_dot  theta_m_ddot] + w_t
Xdot = zeros(6,1);
Xdot(1,:) = theta_t_dot + w_t(1);
Xdot(2,:) = -(c/(m*l^2))*theta_t_dot-1*(g/l)*sin(theta_t) + w_t(2);
Xdot(3,:) = theta_m_dot + w_t(1);
Xdot(4,:) = -1*(9.81/(1.1*l))*sin(theta_m) + w_t(2);
Xdot(5,:) = theta_t_dot_det;
Xdot(6,:) = -(c/(m*l^2))*theta_t_dot_det-1*(g/l)*sin(theta_t_det);
end
