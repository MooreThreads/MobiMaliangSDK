apt --fix-broken install
wget https://mcconline.oss-cn-beijing.aliyuncs.com/software/2023/10/12/musa_2.1.1-Ubuntu-dev_amd64.deb --no-check-certificate
dpkg -i musa_2.1.1-Ubuntu-dev_amd64.deb
rm musa_2.1.1-Ubuntu-dev_amd64.deb
modprobe mtgpu
dpkg -s musa