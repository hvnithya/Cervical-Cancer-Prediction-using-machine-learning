
clear all
close all
clc

addpath('CNN\')
addpath('CNN\util\')


for ijk = 1:1061
    
    %     [filename, pathname]  = uigetfile('Datasets\*.*','Select an image');
    
    
    Input_image1 = imread(['Dataset\IMG (',num2str(ijk),').jpg']);
    
    %     axes(handles.axes1);
    %     imshow(Input_image1);
    %     axis equal;axis off;
    %     title('Input Image');
    
    
    
    Input_image = imresize(Input_image1,[256 256]);
    
    
    BW = im2bw(Input_image);
    
    Ilabel = bwlabel(~BW);
    stat = regionprops(Ilabel,'centroid');
    stat1 = regionprops(Ilabel,'area');
    
    imshow(Input_image); hold on;
    
    for x = 1: numel(stat)
        
        plot(stat(x).Centroid(1),stat(x).Centroid(2),'ro');
        Cval1(x,:) = stat(x).Centroid(1)';
        Cval2(x,:) = stat(x).Centroid(2)';
        Cval(x,:) = [mean(Cval1) mean(Cval2)];
        Aval1(x,:) = stat1(x).Area;
        
    end
    
    
    
    [H,angles] = HOG(Input_image);
    
    
    %     -- Deep Learning method -- %
    
    %     train = imread('2.jpg');
    
    train = imresize(Input_image,[256 256]);
    label = 1:600;
    train_x = double(reshape(train(:,1:600),16,16,600))/255;
    test_x = double(reshape(train(:,1:100),16,16,100))/255;
    train_y = double(label(1:600));
    test_y = double(label(1:100));
    
    rand('state',0)
    
    cnn.layers = {
        struct('type', 'i') %input layer
        struct('type', 'c', 'outputmaps', 6, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %sub sampling layer
        struct('type', 'c', 'outputmaps', 12, 'kernelsize', 5) %convolution layer
        struct('type', 's', 'scale', 2) %subsampling layer
        };
    
    opts.alpha = 1;
    
    opts.batchsize = 50;
    
    opts.numepochs = 1;
    opts.alg = 'adam';  
    
    cnn = cnnsetup(cnn, train_x, train_y);
    
    cnn = cnntrain(cnn, train_x, train_y, opts);
    
    [er, bad] = cnntest(cnn, test_x, test_y);
    
    %plot mean squared error
    
    Features = [cnn.ffW cnn.rL];
    
    if size(Input_image,3) == 3
        CRIM1 = rgb2gray(Input_image);
        
    else
        CRIM1 = (Input_image);
        
    end
    
    inImg = (CRIM1);
    
    filtDims = [2 3];
    imgSize=size(inImg);
    filtDims=filtDims+1-mod(filtDims,2);
    filt=zeros(filtDims);
    nNeigh=numel(filt)-1;
    
    iHelix=snailMatIndex(filtDims);
    filtCenter=ceil((nNeigh+1)/2);
    iNeight=iHelix(iHelix~=filtCenter);
    
    filt(filtCenter)=1;
    filt(iNeight(1))=-1;
    sumLBP=zeros(imgSize);
    
    for i=1:length(iNeight)
        
        currNieghDiff=filter2(filt, inImg, 'same');
        sumLBP=sumLBP+2^(i-1)*(currNieghDiff>0);
        
        if i<length(iNeight)
            
            filt( iNeight(i) )=0;
            filt( iNeight(i+1) )=-1;
            
        end
        
    end
    
    LBPimg=sumLBP;
    
    filtDimsR=floor(filtDims/2);
    iNeight(iNeight>filtCenter)=iNeight(iNeight>filtCenter)-1;
    
    zeroPadRows=zeros(filtDimsR(1), imgSize(2));
    zeroPadCols=zeros(imgSize(1)+2*filtDimsR(1), filtDimsR(2));
    
    inImg=cat(1, zeroPadRows, inImg, zeroPadRows);
    inImg=cat(2, zeroPadCols, inImg, zeroPadCols);
    
    imgSize=size(inImg);
    
    neighMat=true(filtDims);
    
    neighMat( floor(nNeigh/2)+1 )=false;
    weightVec= (2.^( (1:nNeigh)-1 ));
    
    LBPimg=zeros(imgSize);
    
    for iRow=( filtDimsR(1)+1 ):( imgSize(1)-filtDimsR(1) )
        for iCol=( filtDimsR(2)+1 ):( imgSize(2)-filtDimsR(2) )
            
            subImg=inImg(iRow+(-filtDimsR(1):filtDimsR(1)), iCol+(-filtDimsR(2):filtDimsR(2)));
            
            diffVec=repmat(inImg(iRow, iCol), [nNeigh, 1])-subImg(neighMat);
            LBPimg(iRow, iCol)= weightVec*(diffVec(iNeight)>0);
            
        end
    end
    
    LBPimg = LBPimg(( filtDimsR(1)+1 ):( end-filtDimsR(1) ),( filtDimsR(2)+1 ):( end-filtDimsR(2) ));
    
    LBPfeature = mean(LBPimg);
    
    LBPval = LBPfeature(1,:);
        
    
    Trainfea3(ijk,:) = [Features H(1:50)' LBPval];
    ijk
    close all
end


save Trainfea3 Trainfea3
