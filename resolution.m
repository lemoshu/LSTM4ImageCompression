x=1:1:7;
m1 = xlsread('resolution.xlsx');
plot(x,m1(9,:),'mp',x,m1(10,:),'bo',x,m1(11,:),'r*',x,m1(12,:),'gx')
hold on;
plot(x,m1(9,:),'m-','LineWidth',2)
hold on;
plot(x,m1(10,:),'b-','LineWidth',2)
hold on;
plot(x,m1(11,:),'r-','LineWidth',2)
hold on;
plot(x,m1(12,:),'g-','LineWidth',2)
xlabel('图片分辨率 ');
ylabel('MS-SSIM');%注意调整
title('MS-SSIM与图片分辨率的关系(BPP=1.5)')
legend('Conv.LSTM,BPP=1.5','Peephole LSTM,BPP=1.5','JPEG,BPP=1.5','JPEG2000,BPP=1.5')
set(gcf,'color','w');
set(gca,'Xticklabel',{'416x160','480x224','544x288','608x352','672x416','736x480','768x512'});
