clear;
%list of images
imglist=importdata('imglist.txt');
%minimum size of face
minsize=20;

%path of toolbox
caffe_path='D:\caffe1\caffe-master\Build\x64\Release\matcaffe';
pdollar_toolbox_path='D:\Matlab\toolbox\toolbox-master'
caffe_model_path='C:\Users\50129\Desktop\facecode\16\MTCNN_face_detection_alignment-master\code\codes\MTCNNv1\model'
addpath(genpath(caffe_path));
addpath(genpath(pdollar_toolbox_path));

%use cpu
caffe.set_mode_cpu();
%gpu_id=0;
%caffe.set_mode_gpu();	
%caffe.set_device(gpu_id);

%three steps's threshold
threshold=[0.6 0.7 0.7]

%scale factor
factor=0.709;

%load caffe models
prototxt_dir =strcat(caffe_model_path,'/det1.prototxt');
model_dir = strcat(caffe_model_path,'/det1.caffemodel');
PNet=caffe.Net(prototxt_dir,model_dir,'test');
prototxt_dir = strcat(caffe_model_path,'/det2.prototxt');
model_dir = strcat(caffe_model_path,'/det2.caffemodel');
RNet=caffe.Net(prototxt_dir,model_dir,'test');	
prototxt_dir = strcat(caffe_model_path,'/det3.prototxt');
model_dir = strcat(caffe_model_path,'/det3.caffemodel');
ONet=caffe.Net(prototxt_dir,model_dir,'test');
faces=cell(0);	
for i=1:length(imglist)
	img=imread(imglist{i});
	%we recommend you to set minsize as x * short side
	%minl=min([size(img,1) size(img,2)]);
	%minsize=fix(minl*0.1)
    tic
    [boudingboxes points]=detect_face(img,minsize,PNet,RNet,ONet,threshold,false,factor);
	toc
    faces{i,1}={boudingboxes};
	faces{i,2}={points'};
	%show detection result
	numbox=size(boudingboxes,1);
	figure;imshow(img)
	hold on; 
	for j=1:numbox
		plot(points(1:5,j),points(6:10,j),'g.','MarkerSize',10);
		r=rectangle('Position',[boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2)],'Edgecolor','g','LineWidth',3);
        face=imcrop(img,[boudingboxes(j,1:2) boudingboxes(j,3:4)-boudingboxes(j,1:2)]);
        imwrite(face,strcat('C:\Users\50129\Desktop\database2\',num2str(j),'.jpg'));


    end
    hold off; 
end

