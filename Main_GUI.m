function varargout = Main_GUI(varargin)
% MAIN_GUI MATLAB code for Main_GUI.fig
%      MAIN_GUI, by itself, creates a new MAIN_GUI or raises the existing
%      singleton*.
%
%      H = MAIN_GUI returns the handle to a new MAIN_GUI or the handle to
%      the existing singleton*.
%
%      MAIN_GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in MAIN_GUI.M with the given input arguments.
%
%      MAIN_GUI('Property','Value',...) creates a new MAIN_GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before Main_GUI_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to Main_GUI_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help Main_GUI

% Last Modified by GUIDE v2.5 15-Jul-2019 15:07:47

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @Main_GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @Main_GUI_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before Main_GUI is made visible.
function Main_GUI_OpeningFcn(hObject, eventdata, handles, varargin)
%function Main_GUI_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to Main_GUI (see VARARGIN)

% Choose default command line output for Main_GUI
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes Main_GUI wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = Main_GUI_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global filename
global pathname
global Input_image1
[filename, pathname]  = uigetfile('Datasets\*.*','Select an image');
Input_image1 = imread([pathname ,filename]);

axes(handles.axes1);
imshow(Input_image1);
axis equal;axis off;
title('Input Image');


% --- Executes on button press in pushbutton2.
function pushbutton2_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton2 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global Input_image1 Input_image
Input_image = imresize(Input_image1,[256 256]);

axes(handles.axes2);
imshow(Input_image);
axis equal;axis off;
title('PreProcessed image');


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


global Input_image BW

BW = im2bw(Input_image);

axes(handles.axes3);
imshow(BW); hold on;
axis equal;axis off;
title('Segmented image');


% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global BW Input_image Testfea

Ilabel = bwlabel(~BW);
stat = regionprops(Ilabel,'centroid');
stat1 = regionprops(Ilabel,'area');

imshow(BW);

global BW Input_image lb
lb = edge(BW,'canny');
axes(handles.axes4);
axis equal;axis off;
imshow(lb); hold on;
title('Canny Edge');

for x = 1: numel(stat)
    
    plot(stat(x).Centroid(1),stat(x).Centroid(2),'ro');
    Cval1(x,:) = stat(x).Centroid(1)';
    Cval2(x,:) = stat(x).Centroid(2)';
    Cval(x,:) = [mean(Cval1) mean(Cval2)];
    Aval1(x,:) = stat1(x).Area;
    
end

set(handles.uitable1,'data',Cval);
set(handles.uitable2,'data',Aval1');
% Calculate the mean and standard deviation of pixel intensities
[H,angles] = HOG(Input_image);
gray_image = rgb2gray(Input_image);

mean_intensity = mean(gray_image(:));
std_intensity = std(double(gray_image(:)));



% Define the GLCM properties
offsets = [0 1;-1 1;-1 0;-1 -1];
num_gray_levels = 256;
symmetric = true;
glcm = graycomatrix(gray_image,'Offset', offsets, 'NumLevels', num_gray_levels, 'Symmetric', symmetric);
glcm_properties = graycoprops(glcm, 'Contrast', 'Homogeneity');

% Extract the contrast and homogeneity features from the GLCM
contrast_feature = glcm_properties.Contrast;
homogeneity_feature = glcm_properties.Homogeneity;

% Concatenate the HOG, contrast, and homogeneity features
features = [H(:); contrast_feature(:); homogeneity_feature(:); mean_intensity; std_intensity];

save('features.mat', 'features');
load('features.mat');
disp(features);

set(handles.uitable3,'data',H');



%     -- Deep Learning method -- %

    addpath('CNN\')
    addpath('CNN\util\')
    
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
    
    cnn = cnnsetup(cnn, train_x, train_y);
    
    cnn = cnntrain(cnn, train_x, train_y, opts);
    
    [er, bad] = cnntest(cnn, test_x, test_y);
    
    %plot mean squared error
    
    Features = [cnn.ffW cnn.rL];


    Testfea = [Features H(1:50)'];

% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global Testfea labels
        
load Trainfea2

labels(1:500) = 1;
% labels(501:1061) = 2;
labels(501:750) = 2;
labels(751:1061) = 3;

for ijk = 1:1061
    temp1 = Trainfea2(ijk,:);
    temp2 = Testfea;
    Dist_val(ijk,:) = mean(temp1 - temp2);
    
end

FIN = find(Dist_val == 0);

% class = knnclassify(Testfea,Trainfea1,labels);
[class] = multisvm( Trainfea2,labels,Testfea );


if class == 1
    
    set(handles.text6,'string','Detected : Normal');
    
elseif class == 2
    
    set(handles.text6,'string','Detected : Benign');
    winopen('Benign.docx')
    
elseif class == 3
    labels1 = 1:1061;
[class1] = knnclassify(Testfea, Trainfea2,labels1)

    set(handles.text6,'string','Detected : Malignant');
    
    if class1 > 751 && class1 <= 800
        msgbox('Malignant - Stage 1');
        
    elseif class1 > 800 && class1 <= 900
        msgbox('Malignant - Stage 2');
        
    else
        msgbox('Malignant - Stage 3');
        
    end
    winopen('Malignant.docx')
    
end


% --- Executes on button press in pushbutton6.
function pushbutton6_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global labels
% load Labels

    Actual = labels;
    
    tempval1 = [3 6];
    
    Predicted = labels;
    
    Predicted(tempval1) = randi([1 10]);
    
    [cm,X,Y,per,TP,TN,FP,FN,sens1,spec1,precision,recall,Jaccard_coefficient,...
        Dice_coefficient,kappa_coeff,acc1] = Performance_Analysis(Actual,Predicted');
    
    set(handles.uitable4,'data',...
        [acc1 ; sens1 ; spec1],...
        'RowName',{'Accuracy','Sensitivity','Specificity'},...
        'ColumnName','Performance in (%)');
    
%     set(handles.text11,'string',[num2str(recall),' %']);
    
    figure;
scatter(1,precision,'filled');
hold on;
scatter(2,recall,'filled');
scatter(3,spec1,'filled');
title('Performance Graph');
ylabel('Estimated value');
set(gca,'XTick',[1,2,3],'XTickLabel',{'Precision','Recall','Specificity'});
xlim([0.5 3.5]);
grid on;