sig0=0.0001;
etaw0=0.3;
etaa0=0.3;
etat0=0.3;
sig=sig0;
etaw=etaw0;
etaa=etaa0;
etat=etat0;
V=importdata('Vx_Vy_sonar_reading_March1_5000.xls');
Vx=V.data(:,1);
Vy=V.data(:,2);
s_3=V.data(:,3);
s_4=V.data(:,4);
s_5=V.data(:,5);
m=input('Enter no. of neurons in a 2-D lattice    ');
n=input('Enter no. of neurons in a 2-D lattice    ');
for i=1:m
    for j=1:n
        w{i,j}=2*rand(5,1)-1;
        theta{i,j}=2*rand(2,1)-1;
        A{i,j}=2*rand(2,5)-1;
    end
end

for k=1:25000
    r=mod(k,4998)+2;
    u=[s_3(r,1); s_4(r,1); s_5(r,1); Vx(r-1,1); Vy(r-1,1)];
    yd=[Vx(r,1); Vy(r,1)];
    min=999999;
    for i=1:m
        for j=1:n
            if(norm(u-w{i,j})<min)
                min=norm(u-w{i,j});
                iw=i;
                jw=j;
            end
        end
    end
    sum=0;
    sum1=0;
    for i=1:m
        for j=1:n
            dsq=(iw-i)^2 +(jw-j)^2;
            h{i,j}=exp(-dsq/(2*sig^2));
            sum=sum+h{i,j};
            sum1=sum1+h{i,j}*(theta{i,j}+A{i,j}*(u-w{i,j}));
        end
    end
    y=sum1/sum;
    for i=1:m
        for j=1:n
            theta{i,j}=theta{i,j}+etat*h{i,j}*(yd-y)/sum;
            A{i,j}=A{i,j}+etaa*h{i,j}*(yd-y)*(u-w{i,j})';
            w{i,j}=w{i,j}+etaw*h{i,j}*(u-w{i,j});
        end
    end
    e(k)=100*norm(yd-y)/norm(yd);
    ya1(k)=y(1);
    yans1(k)=yd(1);
    ya2(k)=y(2);
    yans2(k)=yd(2);
    sig=sig0*exp(-k/1000);
    etaw=etaw0*exp(-k/1000);
    etaa=etaa0*exp(-k/1000);
    etat=etat0*exp(-k/1000);
end
k=1:25000;
plot(k,e,'r');
pause(5)
subplot(2,1,1);
plot(k,ya1,'r');
hold on;
plot(k,yans1,'b');
xlabel('Iterations ------------->');
ylabel('y  --------->');
legend('y predicted','y desired');
title('Graph for Vx');
hold off
subplot(2,1,2);
plot(k,ya2,'r');
hold on;
plot(k,yans2,'b');
legend('y predicted','y desired');
title('Graph for Vy');
xlabel('Iterations ------------->');
ylabel('y  --------->');
hold off
min=99999999;
ut=[3.83858; 2.332301; 1.785879; 0.013073; 0.149429];
for i=1:m
        for j=1:n
            if(norm(ut-w{i,j})<min)
                min=norm(ut-w{i,j});
                iw=i;
                jw=j;
            end
        end
    end
    sum=0;
    sum1=0;
    for i=1:m
        for j=1:n
            dsq=(iw-i)^2 +(jw-j)^2;
            h{i,j}=exp(-dsq/(2*sig^2));
            sum=sum+h{i,j};
            sum1=sum1+h{i,j}*(theta{i,j}+A{i,j}*(ut-w{i,j}));
        end
    end
    yt=sum1/sum;



        



