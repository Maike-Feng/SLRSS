clear;close all;clc;
%% load data
% name='IndianPines';
name='Salinas';
% name='PaviaU';
% name='Houston2013';
if strcmp(name,'IndianPines')
    load Indian_pines_corrected.mat;
    image=indian_pines_corrected;

elseif strcmp(name,'PaviaU')
    load PaviaU.mat
    image=paviaU;

elseif strcmp(name,'Salinas')
    load Salinas_corrected.mat
    image=salinas_corrected;

elseif strcmp(name,'Houston2013')
    load Houston2013.mat
    image=Houston;
    image=double(image);
end

[w,h,b]=size(image);
nC_list=[30, 50, 70, 100, 200, 300, 400, 500, 600];
for nCi=1:length(nC_list)
    nC=nC_list(nCi);
    %% paramters
    lambda_prime = 0.2;sigma = 5.0; 
    conn8 = 1; 
    results=zeros(w,h,b);
    tic
    disp(['scale= ',num2str(nC),'...']);
    for i=1:b
        disp(['order of the band',num2str(i),'...']);
        pause(0.00005)
        img=image(:,:,i);
        img=reshape(mapminmax(reshape(img,1,w*h),0,255),w,h);
        [labels] = mex_ers(double(img),nC,lambda_prime,sigma);
        results(:,:,i)=labels;
    end
    toc
    savepath='./';
    save_name=[savepath,name,'_255seg',num2str(nC),'.mat'];
    save(save_name,'results')
end 