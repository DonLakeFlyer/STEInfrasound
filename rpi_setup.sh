#!/bin/bash
set -e

# wget https://raw.githubusercontent.com/DonLakeFlyer/STEInfrasound/main/rpi_setup.sh

echo "*** Install tools"
sudo apt install build-essential git cmake -y
git config --global pull.rebase false

echo "*** Create repos directory"
cd ~
if [ ! -d repos ]; then
    mkdir repos
fi
cd ~/repos

echo "*** Clone STEInfrasound"
cd ~/repos
if [ ! -d STEInfrasound ]; then
	git clone --recursive https://github.com/DonLakeFlyer/STEInfrasound.git
fi
cd ~/repos/STEInfrasound
git pull origin main
