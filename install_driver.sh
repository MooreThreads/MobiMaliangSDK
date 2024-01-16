apt --fix-broken install
dpkg -r musa
dpkg -P musa
dpkg -i debs/musa_2.1.1-Ubuntu-dev_amd64.deb
modprobe mtgpu
dpkg -s musa