wget https://mcconline.oss-cn-beijing.aliyuncs.com/software/2023/07/11/sgpu-dkms_1.1.1.deb
dpkg -i sgpu-dkms_1.1.1.deb
rm sgpu-dkms_1.1.1.deb

wget https://mcconline.oss-cn-beijing.aliyuncs.com/software/2023/07/11/mtml_1.5.0.deb
dpkg -i mtml_1.5.0.deb
rm mtml_1.5.0.deb

wget https://mcconline.oss-cn-beijing.aliyuncs.com/software/2023/07/11/mt-container-toolkit_1.5.0.deb
dpkg -i mt-container-toolkit_1.5.0.deb
rm mt-container-toolkit_1.5.0.deb

(cd /usr/bin/musa && ./docker setup $PWD)
cd ~/桌面/mtai_workspace/MobiMaliangSDK

apt-get install git -y
apt-get install docker -y
apt-get install docker.io -y
docker login -u mcc-test@mcconline -p Abc123$%^ registry.mthreads.com
docker pull registry.mthreads.com/mcconline/musa-pytorch-release-public:latest
docker run -id --name mtai_workspace --privileged -e MTHREADS_VISIBLE_DEVICES=all -p 1001:1001 -p 1002:1002 -p 1003:1003 -p 1004:1004 -p 1005:1005 -v ~/桌面/mtai_workspace:/mtai_workspace:rw --shm-size 64G registry.mthreads.com/mcconline/musa-pytorch-release-public:latest
docker exec -it mtai_workspace /bin/bash