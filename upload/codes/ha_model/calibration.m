%--------------------------------------------------------------------------
% Model calibration 
%--------------------------------------------------------------------------
% written by Valerio Pieroni 


%% Parameters

global gamma rho alpha delta 
global I J aa ee da de daf dab dx Ee agrid egrid 
global Delta B

% households
gamma = 1;                        % CRRA, inverse of IES
rho = .07/4;                     % discount rate 5% annualized 
phi = .5;                          % borrowing limit

% idiosyncratic risk logs
ni = -log(0.9)/4; 
sigma_e = 0.2;   
logemean = -0.31;                   % calibrated so that Ee = 1 (continuous states)

% firms   
alpha = .33; 
delta = .05/4;                      % steady state investment rate

% computational
Delta = 1e4; 
I = 40; 
J = 25; 

%% Grids

% % wealth uniform grid
% amax = 60; % higher increase on impact aggregate consumption more people at borrowing constraint      
% amin = -phi;
% a = linspace(amin,amax,I)';
% da = (amax-amin)/(I-1);

% assets grid
amax = 400; 
amin = -phi;
power = 3;
powergrid = zeros(I,1); 
for i = 1:I
powergrid(i) = amin + (amax-amin)*((i - 1)/(I-1))^power; 
end 

agrid = powergrid; 
figure(1)
plot(agrid,'.','MarkerSize',10); 
close(1); 

% stepsize
daf = ones(I,1);
dab = ones(I,1);
daf(1:I-1) = agrid(2:I)-agrid(1:I-1);
dab(2:I) = agrid(2:I)-agrid(1:I-1);
daf(I) = daf(I-1); 
dab(1) = dab(2);
% trapezoidal rule coefficients for integrals
da = 0.5*(dab + daf);
da(1) = 0.5*daf(1); 
da(I) = 0.5*dab(I);
daa = repmat(da,1,J); 

% Ornstein-Uhlenbeck in logs
var = sigma_e^2/(2*ni);
% uniform income risk grid
lemin = logemean - 3*sqrt(var); 
lemax = logemean + 3*sqrt(var); % in equilibrium check that the grid bounds do not cut too much tails of the stationary e distribution 
le = linspace(lemin,lemax,J)';
dle = (lemax-lemin)/(J-1);
% non-uniform log-income risk grid 
e = exp(le); 
egrid = e;                            
% stepsize for continuous approximation
def = ones(J,1);
deb = ones(J,1);
def(1:J-1) = e(2:J) - e(1:J-1);
deb(2:J) = e(2:J) - e(1:J-1);
def(J) = def(J-1); 
deb(1) = deb(2);
% trapezoidal rule coefficients for integrals
de = 0.5*(deb + def);
de(1) = 0.5*def(1); 
de(J) = 0.5*deb(J); 
dee = repmat(de',I,1);

aa = repmat(agrid,1,J); 
ee = repmat(egrid',I,1);  
dx = daa.*dee;

%% Construct matrix summarizing evolution of e

% continuous state 
% Ornstein-Uhlenbeck in logs from Ito's lemma
me = (ni*(logemean - le) + sigma_e^2/2).*e; 
se2 = (sigma_e.*e).^2;
% backward, forward, central coefficients 
Bb =  - min(me,0)./deb + (se2./2).*(def./(0.5*(def + deb).*def.*deb));
Cc =  min(me,0)./deb - max(me,0)./def - (se2./2).*((def + deb)./(0.5*(def + deb).*def.*deb));
Ff = max(me,0)./def + (se2./2).*(deb./(0.5*(def + deb).*def.*deb));
% transition matrix
coeff_c = [(Cc(1) + Bb(1)); Cc(2:J-1); Cc(J) + Ff(J)];
coeff_f = [0; Ff(1:J-1)];  
coeff_b = [Bb(2:J); 0];
T = spdiags([coeff_c coeff_f coeff_b],[0 1 -1],J,J);
% compute distribution
[fe1] = KF1(T); 
cdf = cumsum(fe1.*de); 
Ee = sum(fe1.*e.*de); 


B = sparse(kron(T,eye(I,I)));
spy(B); 
close; 

%% Plot income distribution 

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

% compare discrete approximation 
figure(1)
plot(e,cdf,'LineWidth',1,'Color',grey,'Marker','.','MarkerSize',10,'MarkerEdgeColor',lblue); hold on; 
ylim([0,1]);
legend('Model CDF','Interpreter','tex','Location','southeast'); legend('boxoff'); 
grid on; 

figure(2)
plot(e,fe1); hold on; 

close all; 

%% Functions 

function [fz1] = KF1(P)

global J de

% Solve KF  
b = zeros(J,1);
b(1)= -1;
row = [1,zeros(1,J-1)]';
P(:,1) = row;
% solve the linear system
varf = (P')\b;
% normalize  
fz1 = varf./(varf'*ones(J,1)); 
% to recover psi use varf_i = f_i*dz_i
dxmat = spdiags(de,0,J,J);
fz1 = dxmat\fz1;

end

