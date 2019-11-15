clc;
close all;
clear all;
function lamd=lace(w1,b1,w2,b2,X)
  z1=w1*X'+b1;
  a1=tanh(z1);
  z2=w2*a1+b2;
  a2=tanh(z2);
  lamd=a2;
end
function z=deriofactfun(x)
  z=1- tanh(x).*tanh(x);
end
n=501;
t=linspace(0,200,n);
a=1;b=1;c=1;
g=[a;b;c];
xd=2+t;
yd=3+t;
zd=4+t;
x=xd(:,1:n-1);
y1=yd(:,1:n-1);
z=zd(:,1:n-1);
xd=xd(:,2:n);
yd=yd(:,2:n);
zd=zd(:,2:n);
X=[x' y1' z'];
Xd=[xd' yd' zd'];
E=Xd-X;
scatter3(x,y1,z);
p=10;
w1=rand(p,3);
b1=rand(p,1);
w2=rand(3,p);
b2=rand(3,1);
lamk1=lace(w1,b1,w2,b2,E);
uk1=-g'*lamk1;
dt=0.001;
xk1=X'+dt*(g*uk1);
e2=Xd-(xk1)';
lamk2=lace(w1,b1,w2,b2,e2);
dlamk1=e2+(lamk2)';
y=dlamk1';
nw2=nw1=nb2=nb1=0.085;
for epoch=1:3000
  a0=E';
  for i=1:3
    a0(i,:)=(a0(i,:)-min(a0(i,:)))./(max(a0(i,:))-min(a0(i,:)));
  end
  z1=w1*a0+b1;
  a1=tanh(z1);
  z2=w2*a1+b2;
  a2=z2;
  k1=-g'*lamk1;
  dt=0.001;
  xk1=X'+dt*(g*uk1);
  e2=Xd-(xk1)';
  lamk2=lace(w1,b1,w2,b2,e2);
  dlamk1=e2+(lamk2)';
  y=dlamk1';
  da2=a2-y;
  m=n-1;
  if (rem(epoch,100)==0)    
    (trace(da2*da2'))/m
  end
  dz2=da2;
  dw2=(dz2*a1')./m;
  db2=(sum(dz2,2))./m;
  da1=w2'*dz2;
  dz1=da1.*(deriofactfun(z1));
  dw1=(dz1*a0')./m;
  db1=(sum(dz1,2))./m;
  w2=w2-nw2.*dw2;
  b2=b2-nb2.*db2;  
  w1=w1-nw1.*dw1;
  b1=b1-nb1.*db1;
end 
lamk1=lace(w1,b1,w2,b2,E);
uk1=-g'*lamk1;
dt=0.001;
xk1=X'+dt*(g*uk1);
hold on;
scatter3(xk1(1,:),xk1(2,:),xk1(3,:));