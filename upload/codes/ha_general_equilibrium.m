
%-------------------------------------------------------------------------------------------
%  Heterogeneous agent model in general equilibrium 
%-------------------------------------------------------------------------------------------

% Solve the steady state of a standard heterogeneous agent model using the endogenous grid method (EGM) and 
% fixed point iteration on the Kolmogorov forward equation. For details on model and algorithm see Chapter 3 
% and Chapter 5 in the QuantMacro notes. 
% Written by Valerio Pieroni. 

clear; clc; close all; 

global gamma beta agrid zgrid 
global I J aa zz P


%% Parametrization 

% households 
dr = 0.05;              % discount rate
beta = 1/(1 + dr);      % discount factor 
gamma = 2;              % CRRA coefficient inverse of IES   
phi = 0;                % borrowing limit 
sigma = 0.2;            % variance AR(1) income innovation
rho = 0.9;              % income autoregressive coefficient

% firms 
delta = 0.04;          % depreciation rate 
alpha = 0.33;          % Cobb-douglas capital income share 

% grid size 
I = 80;                % asset grid size
J = 2;                  % income grid size

%% Grids and discretized income process 

% uniform asset grid 
amax = 40; 
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

%% Steady state general equilibrium 

% initialization 
eps = 1e-2;
it = 0;
itmax = 50; 
% guess
rmin = - delta + 1e-8; 
rmax = dr - 1e-8;
r = .5*(rmin+rmax);

tic
while it < itmax

% equilibrium prices   
K = ((r + delta)/(alpha*(L^(1-alpha))))^(1/(alpha-1));  
w = (1-alpha)*L^(-alpha)*K^alpha;

% household optimization
[ap,c] = EGM(r,w);

% stationary distribution  
[faz] = CK(ap);

% market clearing
K1 = sum(faz.*aa,'all');

res = (K1 - K);

% update r using bisection 
if res > eps
    fprintf('Excess Supply, r = %.4f, w = %.4f \n',r,w);
    rmax = r; 
    r = 0.5*(r+rmin);
    it = it + 1;
elseif res < -eps
    fprintf('Excess Demand, r = %.4f, w = %.4f \n',r,w);
    rmin = r; 
    r = 0.5*(r+rmax);
    it = it + 1;
elseif res < eps
    fprintf('Equilibrium Found, (r, w, it) = (%.4f,%.4f,%.4f) \n',r,w,it);
    disp('');
    break
end

disp(res); 

end
toc

%% Plots

close all; 

% plot policy functions
figure(1)
subplot(1,2,1)
plot(agrid,c); 
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('$c(a,z)$','Interpreter','latex','FontSize',14); 
subplot(1,2,2)
plot(agrid,ap-agrid); hold on; plot(agrid,zeros(I,1)); 
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('$\Delta a(a,z)$','Interpreter','latex','FontSize',14); 

figure(2)
fa = sum(faz,2);
abar = linspace(amin,amax,1000); 
fabar = interp1(agrid,fa,abar); 
b = bar(abar,fabar,'hist'); 
%ylim([0,0.05]);
xlim([-0.2,agrid(I)]);
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('Wealth distribution','Interpreter','latex','FontSize',14); 


%% Functions 

function [ap,c] = EGM(r,w)

% solve the euler equation by endogenous grid method 
% output: saving function and consumption function 

global beta gamma I J aa zz P agrid 

% initialization 
ap = zeros(I,J); 
it = 0;
err = 1;
cmin = 1e-4; 

% marginal utility functions 
uc = @(x) x.^(-gamma);
invuc = @(x) x.^(-1/gamma); 

% guess policy function on exogenous saving grid 
s = aa; 
c = w*zz + (1 + r)*s;
cg = max(c,cmin);

while err > 1e-8 && it < 1000

% expected marginal utility, consumption and assets today 
Euc = uc(cg)*P'; 
c = invuc((1 + r)*beta*Euc); 
c = max(c,cmin); 
a = (c + s - w*zz)/(1 + r); 

% a(a',z) is the endogenous grid, i.e. value of assets today that would lead the consumer to have
% a' assets tomorrow if income shock was z. 

% now we want back to the exogenous asset grid. When agrid <= a(1,j) means constrained tomorrow so
% we cannot use the Euler equation. In such cases ap = amin, c(agrid_i,zgrid_j) = (1+r)agrid_i + w*zgrid_j - ap


for j = 1:J

% interpolate over the exogenous grid  
ap(:,j) = (aa(:,j)>a(1,j)).*interp1(a(:,j),s(:,j),aa(:,j),'pchip') + (aa(:,j)<=a(1,j)).*(agrid(1));
c(:,j) = (aa(:,j)>a(1,j)).*interp1(a(:,j),c(:,j),aa(:,j),'pchip')+ (aa(:,j)<=a(1,j)).*((1+r)*aa(:,j) + w*zz(:,j) - ap(:,j)); 

end

% check convergence 
err = max(abs(cg(:) - c(:))); 
it = it + 1; 
cg = c; 

end
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
% the two agrid values (a-1, a+1) closest to ap so that (1-p)a-1 + pa1 = a'
policy = ap(:); 
for s = 1:I*J
% find a' on the asset grid 
[val, ind] = position(agrid,I,policy(s));   
Ia(s,:) = val(1)*(agrid == agrid(ind(1))) + val(2)*(agrid == agrid(ind(2)));    
% find the transitions probabilities
T(s,:) = kron(PP(s,:),Ia(s,:));  
end

% find stationary distribution by fixed point iteration 
faz = (1/(I*J))*ones(I*J,1);
err = 1; 
while err > 1e-8
faz1 = T'*faz;
err = max(abs(faz1-faz));
faz = faz1;
end

faz = reshape(faz,I,J); 

end


function [val, ind] = position(grid,n,x)

% returns the values and the indices of the two points on column vector grid, 
% with dimension n and ascending order, that are closets to the scalar x 
% if x is on the grid return only one nonzero value the first

if min(abs(grid - x)) == 0 % x is on the grid already

ind(1) = find(grid == x);
ind(2) = find(grid == x);
val(1) = 1;
val(2) = 0;    
    
else % x is not on the grid 

% find lower bound on the grid
[~,oldind] = sort([grid; x]);
newind = find(oldind > n);
i = (newind - 1);

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
val(1)=(grid(i+1) - x)/dist;
val(2)=(x - grid(i))/dist;
end

end
end
