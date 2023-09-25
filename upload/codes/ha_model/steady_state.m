
%-------------------------------------------------------------------------------------------
%  Heterogeneous agent model in general equilibrium 
%-------------------------------------------------------------------------------------------
% written by Valerio Pieroni

clear; clc; close all; 

calibration; 

%% Steady state general equilibrium 

global se
se = struct;

% initialization 
eps = 1e-12;
it = 0;
itmax = 50; 
% guess
rmin = 1e-10; 
rmax = rho - 1e-10;
r = .5*(rmin+rmax);
L = Ee; 

% solve equilibrium system 
x0 = r;
options = optimoptions('fsolve','FunctionTolerance',1e-9,'Display','iter');
xstar = fsolve(@(x) steady_state_system(x,L),x0,options);

% store equilibrium 
steady_state_system(xstar,L);

save('equilibrium_steady_state.mat','se');  



%% Plots

close all; 

% plot policy functions
figure(1)
subplot(1,2,1)
surf(egrid,agrid,se.c); ylim([agrid(1),agrid(30)])
ylabel('$a$','Interpreter','latex','FontSize',14); 
zlabel('$c(a,e)$','Interpreter','latex','FontSize',14); 
subplot(1,2,2)
surf(egrid,agrid,se.s); hold on; xlim([egrid(1),egrid(J)]); ylim([agrid(1),agrid(30)]);
ylabel('$a$','Interpreter','latex','FontSize',14); 
zlabel('$s(a,e)$','Interpreter','latex','FontSize',14); 

figure(2)
fa = sum(se.fae.*dee,2);
abar = linspace(amin,amax,1000); 
fabar = interp1(agrid,fa,abar); 
b = bar(abar,fabar,'hist'); 
%ylim([0,0.05]);
xlim([-0.2,agrid(I)]);
xlabel('$a$','Interpreter','latex','FontSize',14); 
ylabel('Wealth distribution','Interpreter','latex','FontSize',14); 




%% Functions 


function [res] = steady_state_system(x,L)

global I J aa ee dx se delta alpha 

r = x; 

% equilibrium prices  
K = ((r + delta)/(alpha*(L^(1-alpha))))^(1/(alpha-1));  
w = (1-alpha)*L^(-alpha)*K^alpha;

% solve HJB
[v,s,c,P] = HJB(w,r,zeros(I,J));     

% solve KF
[fae] = KF(P);

% aggregates and market clearing
C = sum(fae.*c.*dx,'all');
Ks = sum(fae.*aa.*dx,'all');  
Ls = sum(fae.*ee.*dx,'all'); 
Y = (K^alpha)*(L^(1-alpha));

res = K - Ks;

% store equilibrium variables
se = struct('w',w,'r',r,'C',C,'K',K,'Y',Y,'L',Ls,...
            'fae',fae,'v',v,'c',c,'s',s,'P0',P); 

end

function [value,adot,C,P] = HJB(w,r,lumpsum)

% solve HJB equation by finite differences implicit upwind

global gamma rho 
global I J aa ee daf dab 
global Delta B 
% global lambda

% Initialization
Vaf = zeros(I,J); 
Vab = zeros(I,J); 
daaf = daf*ones(1,J);
daab = dab*ones(1,J); 
eps_v = 1; it = 0;

uc = @(c) c.^(-gamma);
c = @(dv) max(dv,1e-12).^(-1/gamma);

if gamma == 1
u =  @(c) log(c); 
v0 = log((w.*ee + r.*aa + lumpsum))./rho;
else    
u = @(c) c.^(1-gamma)./(1-gamma);
v0 = ((w.*ee + r.*aa + lumpsum).^(1-gamma)./(1-gamma))./rho;
end
 
while eps_v > 1e-8    

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

eps_v = max(max(norm(v1 - v0)));

if eps_v <= 1e-5
        %disp('Value Function Converged, Iteration = ')
        %disp(it)
        break
elseif it > 50
    disp('Warning: max it')
    break
end

% update
v0 = v1;
it = it + 1;

end

% output
value = v1; adot = w.*ee + r.*aa + lumpsum - C; P = SD + B; 

end


function [psi] = KF(P)

global I J dx

% We need to solve P'psi=0. 
% This is an homogenoeus linear system. Since P' has a zero eigenvalue it is singular. 
% So it is not invertible and the system admits nontrivial solutions. 

% We have I*J equations in I*J unknown
% Trick is to normalize one element of psi fixing one entry 
% we can do so because by imposing that \sum psi dadz = 1 one equation becomes redundant 
b = zeros(J*I,1);
b(1)= .1;
row = [1,zeros(1,J*I-1)]';
P(:,1) = row;
% solve the linear system
varpsi = (P')\b;
% normalize to recover psi 
psi = varpsi./(varpsi'*ones(J*I,1));
% back out density
dxmat = spdiags(dx(:),0,I*J,I*J);
psi = dxmat\psi; 
psi = reshape(psi,I,J);

end






