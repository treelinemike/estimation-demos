% compare evolutions of the 'continuous' and finite difference representations of the pendulum equation 

% restart
close all; clear all; clc;
rng(1234,'twister');

% define parameters of physical system in a structure
% that we can pass through the ODE solver to the update function
sysParams = [];
sysParams.m = 2;
sysParams.l = 1;
sysParams.c = 1;
sysParams.g = 9.81;
COV_w_true = [0.00002^2 0; 0 0.0002^2];

% initial conditions (state vector: [theta theta_dot]' stacked as: [stochastic truth; undamped model propagation; deterministic truth])
theta_0     = 25*pi/180;      % [rad]
theta_dot_0 = 0;              % [rad/s]
X0 = [theta_0 theta_dot_0]';  % [rad rad/s ]'
X = X0;

% simulation time
t0 = 0;
tf = 20;

% CONTINUOUS PARAMETERS
dt = 0.01;     % [s] timestep size for continuous system
opts = odeset('RelTol',1e-8,'AbsTol',1e-12);

% DISCRETE PARAMETERS
h = 0.0001;   % sampling period for discrete system

% continuous data storage
time = [t0];
x_c = [X0];

% run continuous simulation
for t = t0:dt:(tf-dt)
    
    % calculate timestep for ODE solving
    odeTime = [t t+dt];
    
    % propagate state
    [T,X] = ode45(@(t,X) propDynamics(t,X,sysParams),odeTime,X,opts);
        
    % store results from this timestep
    X = X(end, :)';
    time(end+1)   = T(end);
    x_c(:,end+1) = X; % note: discarding state values at intermediate timesteps calculated by ode45()
end

% compute number of steps to take for discrete system
N = ceil(tf/h);

% discrete data storage
x = X0;  % initial coniditons
x_d = zeros(2,N+1);
x_d(:,1) = x;

% run discrete simulation (note: nonlinear state transition function)
for i = 1:N
    
    % draw noise vector
    w = mvnrnd([0 0]',COV_w_true,1)';
    
    x_next = zeros(size(x));
    x_next(1) = x(1)+h*x(2);
    x_next(2) = (1- (sysParams.c*h/(sysParams.m*sysParams.l^2)))*x(2) - (sysParams.g*h/sysParams.l)*sin(x(1));  
    x = x_next + w;
    x_d(:,i+1) = x;  
end

%% plot results
figure;
ax = subplot(2,1,1);
hold on; grid on;
plot(time,x_c(1,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot((0:N)*h,x_d(1,:),'-','LineWidth',1,'Color',[0 0 0.8]);
xlabel('\bfTime [s]');
ylabel('\bfAngular Position [rad]');
xlim([0 max(time)]);
legend('Continuous',sprintf('Discrete \\Deltat = %0.4fs',h));

ax(end+1) = subplot(2,1,2);
hold on; grid on;
plot(time,x_c(2,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot((0:N)*h,x_d(2,:),'-','LineWidth',1,'Color',[0 0 0.8]);
xlabel('\bfTime [s]');
ylabel('\bfAngular Velocity [rad/s]');
xlim([0 max(time)]);
linkaxes(ax,'x');

%% function to propagate state for ODE solver (continuous system)
function  Xdot = propDynamics(t,X,sysParams)

% recover paramters
m = sysParams.m;
l = sysParams.l;
c = sysParams.c;
g = sysParams.g;

% deconstruct state vector
theta = X(1);
theta_dot = X(2);

% construct Xdot from differential equation
% note:     X    = [theta      theta_dot]
% therefore Xdot = [theta_dot  theta_ddot] + w_t
Xdot = zeros(2,1);

% stochastic truth
Xdot(1) = theta_dot;
Xdot(2) = -(c/(m*l^2))*theta_dot -(g/l)*sin(theta);
end
