x=0.125:0.125:2;
m1 = xlsread('msssim.xlsx');
plot(x,m1(1,:),'mp',x,m1(2,:),'bo',x,m1(3,:),'r*',x,m1(4,:),'gx')
hold on;
plot(x,m1(1,:),'m-','LineWidth',2)
hold on;
plot(x,m1(2,:),'b-','LineWidth',2)
hold on;
plot(x,m1(3,:),'r-','LineWidth',2)
hold on;
plot(x,m1(4,:),'g-','LineWidth',2)
xlabel('Bit Per Pixel(BPP) ');
ylabel('MS-SSIM');%×¢Òâµ÷Õû
legend('Conv.LSTM','Peephole LSTM','JPEG','JPEG2000')
set(gcf,'color','w');
