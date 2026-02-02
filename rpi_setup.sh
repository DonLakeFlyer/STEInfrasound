#!/bin/bash
set -e

# wget https://raw.githubusercontent.com/DonLakeFlyer/STEInfrasound/main/rpi_setup.sh

echo "*** Update system"
sudo apt update
sudo apt full-upgrade -y

echo "*** Install tools"
sudo apt install build-essential git cmake -y
pip install -r requirements.txt
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
