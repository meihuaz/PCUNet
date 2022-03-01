## PCUNet: An Encoder-Decoder Network for 3D Point Cloud Completion

#### 1) environment:  
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch  
pip install future  
pip install pyyaml  
pip install open3d==0.8  
pip install tensorpack==0.8.9  
pip install lmdb   
conda install matplotlib  

install pointnet2  
cd utils/Pointnet2_PyTorch/pointnet2  
python setup.py install  

cd utils/emd   
python setup.py install   

cd utils/ChamferDistancePytorch/chamfer3D  
python setup.py install 

#### 2) Dataset: 
We use PCN dataset (https://drive.google.com/open?id=1M_lJN14Ac1RtPtEQxNlCV9e8pom3U6Pa) and kitti dataset 
(https://drive.google.com/open?id=1M_lJN14Ac1RtPtEQxNlCV9e8pom3U6Pa). 

Run lmdb_serializer.py to preprocess the dataset.

#### 3) Training
1. Run `python train.py` to train the neural network using PCN dataset.
 

#### 4) Testing
1. Run `python test.py` to test the neural network using PCN dataset.




