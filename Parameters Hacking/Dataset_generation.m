clear all, close all, clc
 %chaotic values
sigma = 5;
beta = 28;
r = 8/3;
Parameters_Vector = [sigma;beta;r];
Base_Vector = [sigma;beta;r];
dt = 0.001;
tspan = dt:dt:15;

x0 = [1;1;1];


S=[];
L=[];
number_dataset=100;

for j=1:number_dataset
    [t, x] = ode45(@(t,x)lorenzi(t,x,Parameters_Vector),tspan,x0);
    S(j,:)=[x(:,1)];
    L(j)=Parameters_Vector(1,1);
    Parameters_Vector=Base_Vector+[0.1;0;0]*j;
end
L=L';
S=S';
csvwrite("signals.csv", S)
csvwrite("targets.csv", L)