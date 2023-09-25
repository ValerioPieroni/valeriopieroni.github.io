
%-------------------------------------------------------------------------------------------
%  Heterogeneous agent model equilibrium dynamics
%-------------------------------------------------------------------------------------------
% written by Valerio Pieroni

clear; clc; close all; 


%% steady state 


calibration; 

% initial and final steady state 
load('equilibrium_steady_state.mat');


%% Linearized dynamics 

% non uniform quarterly time grid 
T = 300; 
N = 80;  
tmax = T;
tmin = 0;
power = 3;
powergrid = zeros(N+1,1); % add N+1 to compute dt at the end of the grid
for i = 1:N+1
powergrid(i) = tmin + (tmax-tmin)*((i - 1)/(N+1-1))^power; 
end 
tq = powergrid(1:N); 
dt = powergrid(2:N+1) - powergrid(1:N);
plot(tq,tq,'.');
close all; 

% preallocation partial jacobians
JwK = zeros(N,N); JrK = zeros(N,N); 
JwZ = zeros(N,N); JrZ = zeros(N,N); 
JAr = zeros(N,N); JAw = zeros(N,N); 
J1K = zeros(N,N); J1A = zeros(N,N); 

% steady state sequence add 
K0 = se.K*ones(N,1); 
L0 = se.L*ones(N,1); 
Z0 = ones(N,1); 
w0 = se.w*ones(N,1); 
r0 = se.r*ones(N,1);
Y0 = se.Y*ones(N,1); 
C0 = se.C*ones(N,1);

% sequence of news shocks in each period  
h = 1e-4; 
Ks = se.K*ones(N,N) + h*eye(N); 
Zs =      ones(N,N) + h*eye(N);
ws = se.w*ones(N,N) + h*eye(N);
rs = se.r*ones(N,N) + h*eye(N);


% compute Jacobians by forward accumulation along the DAG 

% first block (inputs' marginal pricing)

FB  = firm(K0, L0, Z0); % "ghost run"

for s = 1:N  

FBt = firm(Ks(1:N,s), L0, Z0);  
JrK(:,s) = (FBt.r   - FB.r)/h;
JwK(:,s) = (FBt.w   - FB.w)/h;

FBt = firm(K0, L0, Zs(1:N,s));
JrZ(:,s) = (FBt.r   - FB.r)/h;
JwZ(:,s) = (FBt.w   - FB.w)/h;

end

% second block (HA)

% partial Js
HB  = households(r0, w0, se, N, dt); % "ghost run"

tic
for s = 1:N  

HBt  = households(rs(1:N,s), w0, se, N, dt);  
JAr(:,s) = (HBt.A   - HB.A)/h;

HBt  = households(r0, ws(1:N,s), se, N, dt); 
JAw(:,s) = (HBt.A   - HB.A)/h;

fprintf(" HA block column =  %d \n", s);

end
clc; 
toc

% total Js forward accumulate
J_AK = JAr * JrK + JAw * JwK; 
J_AZ = JAr * JrZ + JAw * JwZ; 

% last block 

% partial Js
H = mkt_clearing(K0, K0, K0, Y0, C0, dt);  % "ghost run"

for s = 1:N

Ht = mkt_clearing(Ks(1:N,s), K0, K0, Y0, C0, dt);
J1A(:,s) = (Ht.asset_mkt   - H.asset_mkt)/h; 

Ht = mkt_clearing(K0, Ks(1:N,s), K0, Y0, C0, dt);
J1K(:,s) = (Ht.asset_mkt   - H.asset_mkt)/h; 

end

% total Js forward accumulate 
J_1K = J1A * J_AK + J1K; 
J_1Z = J1A * J_AZ; 

% compute total Jacobians of targets H wrt to U, Z 
H_U = J_1K; 
H_Z = J_1Z; 

% compute the GE Jacobian
G = - (H_U\H_Z); 

% save the jacobian 
save('ssj.mat','G');


%% TFP shock 

% TFP shock AR(1) 
Z = 1 + 0.01.*exp(- 0.8*tq);
dZ = Z - 1; 
 
Nx = 30; 

figure(1)
plot(tq,Z); xlim([0,Nx]); 

% General equilibrium Jacobians
GrZ = JrK * G + JrZ; 
GwZ = JwK * G + JwZ; 

% irfs
dK = G   * dZ; 
dr = GrZ * dZ;
dw = GwZ * dZ; 

% solve for the other endogenous variables X = F(K,Z)
% this works only if K, Z are back at steady state at t = T otherwise increase T 
dd = system_dynamics([dK + se.K; se.K], Z, se.L*ones(N,1), N, se, dt);  
dY = dd.Y - se.Y;
dC = dd.C - se.C; 

%% Plots 

close all; 

% colors 
blue = [0 0.09 0.6]; 
lblue = [0 0.45 0.74]; 
llblue = [.47 .68 .82];
dblue = [0,0,0.36]; 
dblue2 = [0 0.29 0.48];
red = [0.9 0 0];
lred = [1 0 0];
dred = [.74 0 0];
green = [0.13 0.6 0.22]; 
black = [0 0 0]; 
grey = [.65 .65 .65];
orange = [.98 .34 .07];
tomato = [.79 .3 .23];
purple = [.43 .2 .29];
dblue3 = [.27 .31 .46];
sand = [.93 .69 .13];

time = tq; 
Nh = N; 
Nx = 20; 

figure(2)

subplot(2,2,1)
plot(time,100*dK(1:Nh)/se.K,'LineWidth',2,'Color',lblue); xlim([0,Nx]); hold on;
plot(time,0*time,'Color',black); 
xlabel('Time','Interpreter','latex','FontName','Times New Roman'); 
ylabel('Deviation (\%)','Interpreter','latex','FontName','Times New Roman'); 
legend('$K_t$','interpreter','latex','Location','southeast'); legend box off; 
%set(gca, 'FontName','Times New Roman','FontSize',14);
set(gca,'FontSize',12);

subplot(2,2,2)
plot(time,100*dC(1:Nh)/se.C,'LineWidth',2,'Color',lblue); xlim([0,Nx]); hold on;
plot(time,100*dY(1:Nh)/se.Y,'LineWidth',2,'Color',dblue); hold on; 
plot(time,0*time,'Color',black); 
xlabel('Time','Interpreter','latex','FontName','Times New Roman'); 
ylabel('Deviation (\%)','Interpreter','latex','FontName','Times New Roman'); 
legend('$C_t$','$Y_t$','interpreter','latex'); legend box off; 
%set(gca, 'FontName','Times New Roman','FontSize',14);
set(gca,'FontSize',12);

subplot(2,2,3)
plot(time,100*dr(1:Nh),'LineWidth',2,'Color',lblue);  xlim([0,Nx]); hold on;
plot(time,0*time,'Color',black); 
xlabel('Time','Interpreter','latex','FontName','Times New Roman'); 
ylabel('Deviation (\%)','Interpreter','latex','FontName','Times New Roman'); 
legend('$r_t$','interpreter','latex'); legend box off; 
%set(gca, 'FontName','Times New Roman','FontSize',14);
set(gca,'FontSize',12);

subplot(2,2,4)
plot(time,100*dw(1:Nh)/se.Y,'LineWidth',2,'Color',lblue);  xlim([0,Nx]); hold on;
plot(time,0*time,'Color',black); 
xlabel('Time','Interpreter','latex','FontName','Times New Roman'); 
ylabel('Deviation (\%)','Interpreter','latex','FontName','Times New Roman'); 
legend('$w_t$','interpreter','latex'); legend box off; 
%set(gca, 'FontName','Times New Roman','FontSize',14);
set(gca,'FontSize',12);



%% Functions 

% define blocks of the model
% ---------------------------------------------------

function  [y] = firm(K,L,Z)

global alpha delta

r = alpha.*Z.*((K./L).^(alpha-1)) - delta; 
w = (1 - alpha).*Z.*((K./L).^alpha); 
Y = Z.*(K.^alpha).*(L.^(1 - alpha)); 

y = struct('r',r,'w',w,'Y',Y);
end

function  [y] = households(r,w,se,N,dt)

global I J aa dx

% preallocation
vt = zeros(I,J,N);
st = zeros(I,J,N); 
ct = zeros(I,J,N); 
P = cell(1,N); 
faet = zeros(I,J,N); 
A = zeros(N,1);
C = zeros(N,1);  

% solve HJB backward
V = se.v;
for t = N:-1:1
vt(:,:,t) = V;
[v,st(:,:,t),ct(:,:,t),P_t] = HJBB(dt(t),w(t),r(t),zeros(I,J),V);
P{t} = P_t;
V = v;
end

% solve KF forward
[faet] = KFF(se.P0,P,dt,N); 

% aggregate  
for t = 1:N
ff = faet(:,:,t); 
cc = ct(:,:,t); 
A(t) = sum(ff.*aa.*dx,'all');
C(t) = sum(ff.*cc.*dx,'all');
end

y = struct('A',A,'C',C); 
end

function  [y] = mkt_clearing(A,K,Kp,Y,C,dt)

global delta

asset_mkt = A - K;
goods_mkt = Y - C - ( (Kp - K)./dt + (delta.*K)); 
y = struct('asset_mkt',asset_mkt,'goods_mkt',goods_mkt);
end

function [yy] = system_dynamics(K,Z,L,N,se,dt)  

% this function combines all the blocks

B1 = firm(K(1:N), L, Z); 
B2 = households(B1.r, B1.w, se, N, dt); 
B3 = mkt_clearing(B2.A, K(1:N), K(2:N+1), B1.Y, B2.C, dt);

yy = struct('r',B1.r,'w',B1.w,'Y',B1.Y,'C',B2.C,'A',B2.A,'asset_mkt',B3.asset_mkt,'goods_mkt',B3.goods_mkt); 

end 


% functions to solve the HA block of the model
% ---------------------------------------------------


function [value,adot,C,P] = HJBB(Delta,w,r,lumpsum,v0)

% solve HJB equation backward from a terminal condition using finite difference implicit method 

global gamma rho 
global I J aa ee daf dab
global B

% Initialization
Vaf = zeros(I,J); 
Vab = zeros(I,J); 
daaf = daf*ones(1,J);
daab = dab*ones(1,J); 

uc = @(c) c.^(-gamma);
c = @(dv) max(dv,1e-12).^(-1/gamma);

if gamma == 1
u =  @(c) log(c); 
else    
u = @(c) c.^(1-gamma)./(1-gamma);
end

% compute numerical derivatives and impose the state constraints
Vaf(1:I-1,:) = (v0(2:I,:)-v0(1:I-1,:))./(aa(2:I,:) - aa(1:I-1,:));
Vaf(I,:) = uc(w.*ee(I,:) + r.*aa(I,:) + lumpsum(I,:)); % will not be used
Vab(2:I,:) = (v0(2:I,:)-v0(1:I-1,:))./(aa(2:I,:) - aa(1:I-1,:));
Vab(1,:) = uc(w.*ee(1,:) + r.*aa(1,:) + lumpsum(1,:));      
  
% compute consumption and saving
cf = c(Vaf);  
sf = w.*ee + r.*aa + lumpsum - cf;  

cb = c(Vab); 
sb = w.*ee + r.*aa + lumpsum - cb; 
    
% upwind scheme: makes a choice of forward or backward differences based on the sign of the state drift
If = sf > 1e-12; 
Ib = sb < -1e-12; 
I0 = (1-If-Ib);  

% Construct HJB in matrix form  
C = cf.*If + cb.*Ib + (w*ee + r.*aa + lumpsum).*I0;
U = u(C);
U = U(:);
v0_vec = v0(:);

X = - min(sb,0)./daab;
Y = - max(sf,0)./daaf + min(sb,0)./daab;
Z = max(sf,0)./daaf;

% construct matrix in sparse form by filling in coefficients on V
ud = reshape([Z(1:I-1,:); zeros(1,J)],[],1);
ld = reshape([zeros(1,J); X(2:I,:)],[],1);
cd = Y(:);
ud = [0; ud(1:end-1)];  
ld = [ld(2:end); 0];
SD = spdiags([cd ud ld],[0 1 -1],I*J,I*J);

% solve the system and update the value
A = (1/Delta+rho)*speye(I*J)-(SD + B);
b = U  + v0_vec./Delta;
v1 = A\b;

v1 = reshape(v1,I,J);

% output
value = v1; adot = w.*ee + r.*aa + lumpsum - C; P = SD + B; 

end



function [faet] = KFF(P_se,Pmat,dt,N)

global I J dx 

% preallocation
faet = zeros(I*J,N);
dist = zeros(I*J,N+1);
dxmat = spdiags(dx(:),0,I*J,I*J);

% normalize one element
b = zeros(I*J,1);
b(1)= .1;
row = [1,zeros(1,I*J - 1)]';
P_se(:,1) = row;
% solve the linear system
varpsi = (P_se')\b;
% normalize to recover ftilde = psi_i*da_i 
f = varpsi./(varpsi'*ones(I*J,1));
ftilde = reshape(f,I,J); 

% initial condition
dist(:,1) = ftilde(:);
% solve KF equations
for t = 1:N
P_t = Pmat{t}';
dist(:,t+1) = (speye(I*J) - P_t*dt(t))\dist(:,t); 
end 
% find density
for t = 1:N
faet(:,t) = dxmat\dist(:,t);
end  

faet = reshape(faet,I,J,N); 

end
