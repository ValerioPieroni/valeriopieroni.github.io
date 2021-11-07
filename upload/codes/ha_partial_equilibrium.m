
%-------------------------------------------------------------------------------------------
%  Heterogeneous agent model in partial equilibrium 
%-------------------------------------------------------------------------------------------

% Solve the steady state of a standard heterogeneous agent model using discrete value function
% iteration (VFI). The goal is to learn discrete VFI with shocks and how to compute a fixed point 
% of the Chapman-Kolmogorov equation. For details on model and algorithm see Chapter 3 and Chapter 5 in the QuantMacro notes.  

% Written by Valerio Pieroni. 


clear; clc; close all; 

global gamma beta agrid zgrid 
global I J aa zz P


%% Parametrization 

% households 
dr = 0.04;              % discount rate
beta = 1/(1 + dr);      % discount factor 
gamma = 1;              % CRRA coefficient inverse of IES   
phi = 0;                % borrowing limit 
sigma = 0.2;            % variance AR(1) income innovation
rho = 0.9;              % income autoregressive coefficient

% firms 
delta = 0.04;          % depreciation rate 
alpha = 0.33;          % Cobb-douglas capital income share 

% grid size 
I = 1000;                % asset grid 
J = 2;                 % income grid 

%% Grids and discretized income process 

% uniform asset grid 
amax = 15; 
amin = - phi;
agrid = zeros(I,1); 
for i = 1:I 
agrid(i) = amin+(i-1)*(amax-amin)/(I-1);
end 

% for now just use a two point markov process to approximate the log AR(1) process 
% log z' = \rho ln z + \sigma \nu  where \nu \sim N(0,1)

% income grid 
sigmaly2 = (sigma^2)/(1 - rho^2); 
den = exp(1 - sqrt(sigmaly2)) + exp(1 + sqrt(sigmaly2)); 
zl = (2*exp(1-sqrt(sigmaly2)))/den; 
zh = (2*exp(1+sqrt(sigmaly2)))/den;
zgrid = [zl; zh];

% combined grids 
aa = repmat(agrid,1,J); 
zz = repmat(zgrid',I,1);

% transition matrix 
Phi = 1 + log(2/den);
kappa = ((sigma^2)*(1+rho) - (Phi^2)*(1-rho))/(2*(sigma^2)); 
P = [kappa 1-kappa; 1-kappa kappa]; 

% initial guess  
p = [.6; .4]; 
test = 1;
% stationary distribution 
while test > 1e-6    
p1 = P'*p;
test = max(abs(p1 - p));
p = p1;        
end

% labor supply
L = zgrid'*p;

%% Partial equilibrium 

w = 1; 
r = .03;

tic
% household optimization
[v,ap,c] = VFI(r,w);
toc

tic
% stationary distribution  
[faz] = CK(ap);
toc

%% Plots

close all; 

% plot policy functions
figure(1)
subplot(2,2,1)
plot(agrid,c);
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('Consumption $c(a,z)$','Interpreter','latex','FontSize',14); 
subplot(2,2,2)
plot(agrid,ap); hold on; plot(agrid,agrid); 
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('Asset $a''(a,z)$','Interpreter','latex','FontSize',14); 
subplot(2,2,3)
plot(agrid,ap-agrid); hold on; plot(agrid,zeros(I,1)); 
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('Saving $\Delta a(a,z)$','Interpreter','latex','FontSize',14); 
subplot(2,2,4)
plot(agrid,v);
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('Value $v(a,z)$','Interpreter','latex','FontSize',14); 

figure(2)
fa = sum(faz,2);
abar = linspace(amin,amax,1000); 
fabar = interp1(agrid,fa,abar); 
b = bar(abar,fabar,'hist'); 
ylim([0,0.05]);
xlim([-0.2,agrid(I)]);
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('Wealth distribution','Interpreter','latex','FontSize',14); 

% The jiggles in the policy functions and spikes in the distribution function are due to the discretization of the state space. Since we solved 
% the Bellman operator on the grid it is unlikely that the maximum k' is indeed a true maximizer. One could refine the grid around each maximum 
% or solve directly the Bellman optimization numerically. These solutions require interpolation. Moreover, they substantially increase the computational 
% burden and other adjustments to the code should be implemented to speed up computations. Alternatively, the endogenous grid method is a more efficient 
% algorithm.

% The consumption function is concave around the borrowing limit. This implies high MPC for constrained agents and low-wealth unconstrained agents 
% as an increase in their wealth relax the precautionary saving motive. Wealthy households are well insured against income shocks.  

%% Functions 

function [v,ap,c] = VFI(r,w)

% sol the Bellman equation 
% input: equilibrium prices
% output: policy and value functions 

global beta gamma I J P agrid zgrid aa zz

% initialization 
it = 0;
err = 1; 
apmat = repmat(agrid',I*J,1); 
amat = repmat(agrid,J,I); 
zmat = kron(zgrid,ones(I,I));

% utility functions 
if gamma > 1
util = @(x) (x.^(1-gamma))./(1-gamma); 
elseif gamma == 1
util = @(x) log(x); 
end

% compute utility IJ x I matrix 
c = zmat*w +(1+r)*amat - apmat;                                                        
u = double(util(c));
u(c < 0) = - 1e8;

% guess the value function 
v0 = zeros(J,I);

while err > 1e-3 && it < 2000

% expected value
Ev = P*v0;
Ev1 = repmat(Ev(1,:),I,1);
Ev2 = repmat(Ev(2,:),I,1);
Ev = [Ev1; Ev2]; 

% Bellman operator w\o grid refinement
[v1,ind] = max(u + beta*Ev,[],2);
ap = agrid(ind);

% check convergence and update 
v0 = [v0(1,:)'; v0(2,:)'];
err = max(abs(v1 - v0));
v0 = [v1(1:I)'; v1(I+1:I*J)']; 
it = it + 1;  

end

% output
v = [v1(1:I) v1(I+1:I*J)]; 
ap = [ap(1:I) ap(I+1:I*J)]; 
c = zz*w + (1+r)*aa - ap; 

fprintf('Value function converged, (it, err) = (%.0f,%.0e) \n',it,err);

end

function faz = CK(ap)

% iterate over Chapman-Kolmogorov equation to find the stationary 
% distribution for (a,z)

global I J P agrid 

% construct the transition matrix from each (a,z) to each (a',z')
T = zeros(I*J,I*J); 
Ia = zeros(I*J,I);
PP = kron(P,ones(I,1));
% To find a' on the asset grid. Given j split the probability 1 of the move from a to a' to
% the two agrid values (a-1, a+1) closest to ap so that (1-p)a-1 + pa1 = a'. This is useful 
% for EGM, in our case by construction we are always on the grid when moving from a to a' 
policy = ap(:); 
for s = 1:I*J
% find a' on the asset grid   
[val, ind] = position(agrid,I,policy(s));   
Ia(s,:) = val(1)*(agrid == agrid(ind(1))) + val(2)*(agrid == agrid(ind(2)));    
% find the transitions probabilities
T(s,:) = kron(PP(s,:),Ia(s,:));  
end
T = sparse(T); 

% find stationary distribution by fixed point iteration 
faz = (1/(I*J))*ones(I*J,1);
err = 1; 
while err > 1e-6
faz1 = T'*faz;
err = max(abs(faz1-faz));
faz = faz1;
end

faz = reshape(faz,I,J); 

fprintf('Stationary distribution found \n');

end


function [val, ind] = position(grid,n,x)

% returns the values and the indices of the two points on column vector grid, 
% with dimension n and ascending order, that are closets to the scalar x 
% if x is on the grid return only one nonzero value

if min(abs(grid - x)) == 0 % x is on the grid already

ind(1) = find(grid == x);
ind(2) = find(grid == x);
val(1) = 1;
val(2) = 0;    
    
else % x is not on the grid 

% find lower bound on the grid
[~,ind] = sort([grid; x]);
temp = find(ind > n);
i = (temp - 1);

if ((i+1)>n)
ind(1) = n;
ind(2) = n;
val(1) = 1;
val(2) = 0;
elseif (i==0)
ind(1) = 1;
ind(2) = 1;
val(1) = 1;
val(2) = 0;
else
ind(1) = i;
ind(2) = i+1;
dist = grid(i+1)-grid(i);
val(2)=(x - grid(i))/dist;
val(1)=(grid(i+1)- x)/dist;
end

end
end
