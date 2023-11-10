wget https://mcconline.oss-cn-beijing.aliyuncs.com/software/2023/07/11/sgpu-dkms_1.1.1.deb --no-check-certificate
dpkg -i sgpu-dkms_1.1.1.deb
rm sgpu-dkms_1.1.1.deb

wget https://mcconline.oss-cn-beijing.aliyuncs.com/software/2023/07/11/mtml_1.5.0.deb --no-check-certificate
dpkg -i mtml_1.5.0.deb
rm mtml_1.5.0.deb

wget https://mcconline.oss-cn-beijing.aliyuncs.com/software/2023/07/11/mt-container-toolkit_1.5.0.deb --no-check-certificate
dpkg -i mt-container-toolkit_1.5.0.deb
rm mt-container-toolkit_1.5.0.deb

(cd /usr/bin/musa && ./docker setup $PWD)

apt-get install git -y
apt-get install docker -y
apt-get install docker.io -y
docker login -u 账号 -p 密码 registry.mthreads.com
docker pull registry.mthreads.com/mcconline/musa-pytorch-release-public:latest

cd ~/Desktop/mtai_workspace/MobiMaliangSDK
cd ~/桌面/mtai_workspace/MobiMaliangSDK