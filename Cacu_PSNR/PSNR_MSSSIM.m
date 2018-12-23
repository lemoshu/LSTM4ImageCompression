a=imread('F:\pytorch-image-comp-rnn\test_pic\kodim24.png');%读取未压缩照片
m=24;
%%
for i=0:15
imageName=strcat(num2str(i),'.png');%需要把图片的name全部命名为0.png样式
b = imread(imageName);
%b=imread('F:\image_encoder\kodim08\image_num2str(i).png');
[PSNR, MSE]=psnr(a,b);%这个下标经常有点问题
img1=rgb2gray(a);%转换为灰度图
img2=rgb2gray(b);
MSSSIM = msssim(img1, img2);
%% 结果存储
resPSNR(m,i+1)=PSNR;%是kodim第几张就在行上改数字
resMSSSIM(m,i+1)=MSSSIM;%是kodim第几张就在行上改数字
end