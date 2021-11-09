
%-------------------------------------------------------------------------------------------
%  OLG - Life-cycle model with idiosyncratic income risk in partial equilibrium 
%-------------------------------------------------------------------------------------------

% Solve the steady state of an OLG-LC model with idiosyncratic income risk using the endogenous grid method (EGM) and 
% fixed point iteration on the Chapman-Kolmogorov forward equation. 
% For a description of the model see Chapter 5 in the QuantMacro notes.

% Written by Valerio Pieroni.   

clear; clc; close all; 

global gamma beta agrid zgrid 
global I J T aa zz P


%% Parametrization 

% households 
dr = 0.04;              % discount rate
beta = 1/(1 + dr);      % discount factor 
gamma = 2;              % CRRA coefficient inverse of IES   
phi = 0;                % borrowing limit 
sigma = 0.2;            % variance AR(1) income innovation
rho = 0.9;              % income autoregressive coefficient

% grid size 
I = 100;                 % asset grid size
J = 2;                  % income grid size

% life-cycle
T = 61;                 % finite horizon t = T leave the model at age 85, t = 1 enter at age 25  


%% Grids and discretized income process 

% uniform asset grid 
amax = 30; 
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

% transition matrix 
Phi = 1 + log(2/den);
kappa = ((sigma^2)*(1+rho) - (Phi^2)*(1-rho))/(2*(sigma^2)); 
P = [kappa 1-kappa; 1-kappa kappa]; 

% combined grids 
aa = repmat(agrid,1,J); 
zz = repmat(zgrid',I,1);

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
r = .04;

tic
% household optimization
[ap,c] = EGM(r,w);

% stationary distribution 
[faz] = CK(ap,p);
toc

%% Plots

close all; 

% colors 
blue = [0 0.09 0.6]; 
lblue = [0 0.45 0.74]; 
red = [0.9 0 0];
lred = [1 0 0];
dred = [.74 0 0];
green = [0.13 0.6 0.22]; 
black = [0 0 0]; 
grey = [.65 .65 .65];
orange = [.98 .34 .07];

% plot policy functions

avec = [1 21 41 61]; 
age = (20:20+T-1); 

figure(1)
subplot(1,2,1)
plot(agrid,c(:,1,1),'LineWidth',1,'Color',blue); hold on; 
plot(agrid,c(:,1,21),'LineWidth',1,'Color',orange);
plot(agrid,c(:,1,41),'LineWidth',1,'Color',green);
plot(agrid,c(:,2,1),'LineWidth',1,'Color',blue); 
plot(agrid,c(:,2,21),'LineWidth',1,'Color',orange);
plot(agrid,c(:,2,41),'LineWidth',1,'Color',green);
%plot(agrid,c(:,1,61));
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('$c_t(a,z)$','Interpreter','latex','FontSize',14); 
subplot(1,2,2)
plot(agrid,ap(:,1,1)-agrid,'LineWidth',1,'Color',blue); hold on; ylim([-.5,.5]);
plot(agrid,ap(:,1,21)-agrid,'LineWidth',1,'Color',orange);
plot(agrid,ap(:,1,41)-agrid,'LineWidth',1,'Color',green);
plot(agrid,ap(:,2,1)-agrid,'LineWidth',1,'Color',blue); 
plot(agrid,ap(:,2,21)-agrid,'LineWidth',1,'Color',orange);
plot(agrid,ap(:,2,41)-agrid,'LineWidth',1,'Color',green);
%plot(agrid,ap(:,1,61)-agrid);
plot(agrid,zeros(I,1),'--','Color',black); 
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('$\Delta a_t(a,z)$','Interpreter','latex','FontSize',14); 

figure(2)
fa1 = sum(faz(:,:,1),2);
fa21 = sum(faz(:,:,21),2);
fa41 = sum(faz(:,:,41),2);
fa61 = sum(faz(:,:,61),2);
plot(agrid,fa1,'LineWidth',1,'Color',blue); hold on; ylim([0,0.8]); xlim([amin,20]);
plot(agrid,fa21,'LineWidth',1,'Color',orange);
plot(agrid,fa41,'LineWidth',1,'Color',green);
plot(agrid,fa61,'LineWidth',1,'Color',lblue);
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('$p_t(a)$','Interpreter','latex','FontSize',14); 


%% Functions 

function [ap,c] = EGM(r,w)

% solve the euler equation by endogenous grid method 
% output: saving function and consumption function 

global beta gamma I J aa zz P agrid T

% Initialization 
c = zeros(I,J,T); 
ap = zeros(I,J,T);  
cmin = 1e-4; 

% marginal utility functions 
uc = @(x) x.^(-gamma);
invuc = @(x) x.^(-1/gamma); 

% policy function on exogenous saving grid at t = T
s = aa; 
ap(:,:,T) = zeros(I,J); 
c(:,:,T) = w*zz + (1 + r)*s;
c(:,:,T) = max(c(:,:,T),cmin);

% solve backward 
for t = T-1:-1:1

% expected marginal utility, consumption and assets today 
Euc = uc(c(:,:,t+1))*P'; 
c(:,:,t) = invuc((1 + r)*beta*Euc); 
c(:,:,t) = max(c(:,:,t),cmin); 
a = (c(:,:,t) + s - w*zz)/(1 + r); 

% a(a',z) is the endogenous grid, i.e. value of assets today that would lead the consumer to have
% a' assets tomorrow if income shock was z. 

% now we want back to the exogenous asset grid. When agrid <= a(1,j) means constrained tomorrow so
% we cannot use the Euler equation. In such cases ap = amin, c(agrid_i,zgrid_j) = (1+r)agrid_i + w*zgrid_j - ap


for j = 1:J

% interpolate over the exogenous grid  
ap(:,j,t) = (aa(:,j)>a(1,j)).*interp1(a(:,j),s(:,j),aa(:,j),'pchip') + (aa(:,j)<=a(1,j)).*(agrid(1));
c(:,j,t) = (aa(:,j)>a(1,j)).*interp1(a(:,j),c(:,j,t),aa(:,j),'pchip')+ (aa(:,j)<=a(1,j)).*((1+r)*aa(:,j) + w*zz(:,j) - ap(:,j,t)); 

end


end
end



function faz = CK(ap,p)

% solve Chapman-Kolmogorov equation forward to find the sequence of 
% distributions for (a,z)

global I J T P agrid

% construct the transition matrix from each (a,z) to each (a',z')
TT = zeros(I*J,I*J,T); 
faz = zeros(I*J,T); 
Ia = zeros(I*J,I,T);
PP = kron(P,ones(I,1));

policy = zeros(I*J,T); 
for t = 1:T
policy(:,t) = reshape(ap(:,:,t),I*J,1);
end 

% To find a' on the asset grid. Given j split the probability 1 of the move from a to a' to
% the two agrid values (a-1, a+1) closest to ap so that (1-p)a-1 + pa1 = a'
for t = 1:T
for s = 1:I*J
% find a' on the asset grid 
[val, ind] = position(agrid,I,policy(s,t));   
Ia(s,:,t) = val(1)*(agrid == agrid(ind(1))) + val(2)*(agrid == agrid(ind(2)));    
% find the transitions probabilities
TT(s,:,t) = kron(PP(s,:),Ia(s,:,t));  
end
end

% distribution of newborns (if all enter economy with 0 assets)
faz(1,1) = p(1); 
faz(I+1,1) = p(2); 
% solve forward distribution sequence  
for t = 1:T-1
faz(:,t+1) = TT(:,:,t)'*faz(:,t);
end

faz = reshape(faz,I,J,T); 

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



