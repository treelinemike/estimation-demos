% compare evolutions of the 'continuous' and finite difference representations of the pendulum equation 

% restart
close all; clear all; clc;

% define parameters of physical system in a structure
% that we can pass through the ODE solver to the update function
sysParams = [];
sysParams.m = 2;
sysParams.l = 1;
sysParams.c = 1;
sysParams.g = 9.81;

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
h = 0.001;   % sampling period for discrete system

% continuous data storage
time = [t0];
data = [X0];

% run continuous simulation
for t = t0:dt:(tf-dt)
    
    % calculate timestep for ODE solving
    odeTime = [t t+dt];
    
    % propagate state
    [T,X] = ode45(@(t,X) propDynamics(t,X,sysParams),odeTime,X,opts);
        
    % store results from this timestep
    X = X(end, :)';
    time(end+1)   = T(end);
    data(:,end+1) = X; % note: discarding state values at intermediate timesteps calculated by ode45()
end

% compute state transition matrix
% NOTE: the sine term has been LINEARIZED to produce this!
c1 = 1 + sysParams.c*h/(sysParams.m*sysParams.l^2) + sysParams.g*h^2/sysParams.l;
PHI = [(2 + (sysParams.c*h)/(sysParams.m*sysParams.l^2))/c1, -1/c1; 1 0];

% compute number of steps to take
N = ceil(tf/h);

% discrete data storage
x = [X0(1); X0(1)-(h*X0(2))];  % initial coniditons
x_hist = zeros(2,N+1);
x_hist(:,1) = x;

% run discrete simulation
for i = 1:N
    x = PHI*x;
    x_hist(:,i+1) = x;  
end

figure;
hold on; grid on;
plot(time,data(1,:),'-','LineWidth',1.6,'Color',[0 0.8 0]);
plot((0:N)*h,x_hist(1,:),'--','LineWidth',1.6,'Color',[0 0 0.8]);
xlabel('\bfTime [s]');
ylabel('\bfPosition [rad]');
xlim([0 max(time)]);
legend('Continuous',sprintf('Discrete \\Deltat = %0.4fs',h));

% function to propagate state for ODE solver
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
