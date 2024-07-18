clear all
close all
clc

t0                  = 0;                % --- Initial time
t1                  = 1;                % --- Final time

N                   = 40;               % --- Number of time-steps 

y0                  = 1;                % --- Initial value of unknown function

h                   = (t1 - t0) / N;    % --- Discretization step 


t                   = zeros(1, N);      % --- Initialization of time array
t(1)                = t0;               % --- Initialization of time array 
yEuler              = zeros(1, N);      % --- Initialization of Euler array
yEuler(1)           = y0; 
yModifiedEuler      = zeros(1, N);      % --- Initialization of Modified Euler array
yModifiedEuler(1)   = y0; 
yRK2                = zeros(1, N);      % --- Initialization of RK2 array
yRK2(1)             = y0; 
yRK4                = zeros(1, N);      % --- Initialization of RK4 array
yRK4(1)             = y0; 
for n = 1 : N 
    t(n + 1)                = t(n) + h; 
    
    % --- Euler
    yEuler(n + 1)           = yEuler(n) + h * f(t(n), yEuler(n)); 
    
    % --- Modified Euler
    wn                      = yModifiedEuler(n);
    tn                      = t(n);
    yModifiedEuler(n + 1)   = wn + 0.5 * h * (f(tn, wn) + f(t(n + 1), wn + h * f(tn, wn))); 
    
    % --- RK2
    wn                      = yRK2(n);
    yRK2(n + 1)             = wn + h * f(tn + 0.5 * h, wn + 0.5 * h * f(tn, wn));
    
    % --- RK4
    wn                      = yRK4(n);
    k1                      = h * f(tn, wn);
    k2                      = h * f(tn + 0.5 * h, wn + 0.5 * k1);
    k3                      = h * f(tn + 0.5 * h, wn + 0.5 * k2);
    k4                      = h * f(t(n + 1), wn + k3);
    yRK4(n + 1)             = wn + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4);
end
  
yexact  = 2 * ones(size(t)) + t -exp(-t);

figure(1)
plot(t, yEuler) 
hold on
plot(t, yModifiedEuler, 'g') 
plot(t, yRK2, 'k') 
plot(t, yRK4, 'r') 
plot(t, yexact, 'o')
title('Solutions')
legend('Euler', 'Modified Euler', 'Runge-Kutta 2', 'Runge Kutta 4', 'Exact')

figure(2)
plot(t, 20 * log10(abs(yEuler - yexact))) 
hold on
plot(t, 20 * log10(abs(yModifiedEuler - yexact)), 'g') 
plot(t, 20 * log10(abs(yRK2 - yexact)), 'k') 
plot(t, 20 * log10(abs(yRK4 - yexact)), 'r') 
title('Accuracies') 
legend('Euler', 'Modified Euler', 'Runge-Kutta 2', 'Runge Kutta 4')


