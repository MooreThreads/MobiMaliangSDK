apt --fix-broken install
dpkg -i deb/musa_2.1.1-Ubuntu-dev_amd64.deb
modprobe mtgpu
dpkg -s musa