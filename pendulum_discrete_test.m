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
sysParams.COV_w_true = [0.0001^2 0; 0 0.0002^2];

% initial conditions (state vector: [theta theta_dot]' stacked as: [stochastic truth; undamped model propagation; deterministic truth])
theta_0     = 25*pi/180;      % [rad]
theta_dot_0 = 0;              % [rad/s]
X0 = [theta_0 theta_dot_0]';  % [rad rad/s ]'
X = X0;

% simulation time
t0 = 0;
tf = 20;

% CONTINUOUS PARAMETERS
dt_c = 0.01;     % [s] timestep size for continuous system
opts = odeset('RelTol',1e-8,'AbsTol',1e-12);

% DISCRETE PARAMETERS
dt_d = 0.001;   % sampling period for discrete system

% SAMPLING PARAMETERS
dt_s = 0.01;
Ns = round(dt_s/dt_d);
dt_s = Ns*dt_d;

%% compute undamped and damped frequencies and time constants
omega_n = sqrt(sysParams.g/sysParams.l);
tau_n = 2*pi/omega_n;
c_cr_eq = 2*omega_n;
zeta = (sysParams.c/(sysParams.m*sysParams.l^2))/c_cr_eq;
omega_d = omega_n*sqrt(1-zeta^2);
tau_d = 2*pi/omega_d;

%% CONTINUOUS SIMULATION 
% continuous data storage
time = [t0];
x_c = [X0];

% run continuous simulation
for t = t0:dt_c:(tf-dt_c)
    
    % calculate timestep for ODE solving
    odeTime = [t t+dt_c];
    
    % propagate state
    [T,X] = ode45(@(t,X) propDynamics(t,X,sysParams),odeTime,X,opts);
        
    % store results from this timestep
    X = X(end, :)';
    time(end+1)   = T(end);
    x_c(:,end+1) = X; % note: discarding state values at intermediate timesteps calculated by ode45()
end

%% DISCRETE SIMULATION
% compute number of steps to take for discrete system
N = ceil(tf/dt_d);

% discrete data storage
x = X0;  % initial coniditons
x_d = zeros(2,N+1);
x_d(:,1) = x;
t_d = (0:N)*dt_d;

% run discrete simulation (note: nonlinear state transition function)
x_d = stepDynamics(N,dt_d,x,sysParams);

%% plot results
figure;
ax = subplot(3,1,1);
hold on; grid on;
plot(time,x_c(1,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot(t_d,x_d(1,:),'-','LineWidth',1,'Color',[0 0 0.8]);
xlabel('\bfTime [s]');
ylabel('\bfAngular Position [rad]');
xlim([0 max(time)]);
legend('Continuous (ode45)',sprintf('Discrete \\Deltat = %0.4fs',dt_d));

ax(end+1) = subplot(3,1,2);
hold on; grid on;
plot(time,x_c(2,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot(t_d,x_d(2,:),'-','LineWidth',1,'Color',[0 0 0.8]);
xlabel('\bfTime [s]');
ylabel('\bfAngular Velocity [rad/s]');
xlim([0 max(time)]);

ax(end+1) = subplot(3,1,3);
hold on; grid on;
plot(time,gradient(x_c(2,:),time),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot(t_d,gradient(x_d(2,:),t_d),'-','LineWidth',1,'Color',[0 0 0.8]);
xlabel('\bfTime [s]');
ylabel('\bfAngular Acceleration [rad/s^2]');
xlim([0 max(time)]);
linkaxes(ax,'x');

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

%% function to propagate state via ODE solver ("continuous" dynamics)
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
