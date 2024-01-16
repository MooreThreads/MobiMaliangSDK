dpkg -i debs/sgpu-dkms_1.1.1.deb
dpkg -i debs/mtml_1.5.0.deb
dpkg -i debs/mt-container-toolkit_1.5.0.deb

(cd /usr/bin/musa && ./docker setup $PWD)

apt-get install git -y
apt-get install docker -y
apt-get install docker.io -y
docker pull registry.mthreads.com/mcconline/musa-pytorch-release-public:latest