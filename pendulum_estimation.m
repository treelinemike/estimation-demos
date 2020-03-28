% NOTE: CONSIDER REMOVING LATEX INTERPRETER IN LEGENDS B/C MAC AND SOME
% OLDER VERSIONS OF MATLAB THROW ERRORS WITH THAT

% plot phase portraits for the simple pendulum

% restart
close all; clear all; clc;

% reset random number generator using consistant seed
% although noise is random it will be the same for every run
rng(2374,'twister');

% options
sigma_v = 0.002; % standard deviation of sensor noise

% simulation time parameters
t0 = 0;       % [s] simulation start time
tf = 20;      % [s] simulation end time
dt = 0.1;     % [s] timestep size
opts = odeset('RelTol',1e-8,'AbsTol',1e-12);

% define parameters for cases to analyze
sysParams = [];
sysParams.m = 2;   
sysParams.l = 1;
sysParams.c = 1;
sysParams.g = 9.81;

% initial conditions (state vector: [theta theta_dot]')
theta_t_0     = 25*pi/180;      % [rad]
theta_t_dot_0 = 0;              % [rad/s]
theta_m_0     = 25*pi/180;      % [rad]
theta_m_dot_0 = 0;              % [rad/s]
X0 = [theta_t_0 theta_t_dot_0 theta_m_0 theta_m_dot_0]';  % [rad rad/s rad rad/s]'
X = X0;

% data storage
time = [t0];
data = [X0];

% run simulation
for t = t0:dt:(tf-dt)
    
    % calculate timestep for ODE solving
    odeTime = [t t+dt];
    
    % propigate state
    [T,X] = ode45(@(t,X) propDynamics(t,X,sysParams),odeTime,X,opts);
    X = X(end, :)';  % note: this step is necessary to keep state vector dimensions correct for next call to ode45()
    
    % store results from this timestep
    time(end+1)   = T(end);
    data(:,end+1) = X; % note: discarding state values at intermediate timesteps calculated by ode45()
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

% propagate state
function  Xdot = propDynamics(t,X,sysParams)

% recover paramters
m = sysParams.m;
l = sysParams.l;
c = sysParams.c;
g = sysParams.g; 

% deconstruct state vector
theta_t = X(1);
theta_t_dot = X(2);
theta_m = X(3);
theta_m_dot = X(4);

% construct Xdot from differential equation
% note:     X    = [theta      theta_dot]
% therefore Xdot = [theta_dot  theta_ddot]
Xdot = zeros(4,1);
Xdot(1,:) = theta_t_dot;
Xdot(2,:) = -(c/(m*l^2))*theta_t_dot-1*(g/l)*sin(theta_t);
Xdot(3,:) = theta_m_dot;
Xdot(4,:) = -1*(9.81/(1.1*l))*sin(theta_m);
end
