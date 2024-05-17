#!/bin/bash
# Run `source setup.sh`
# yum install -y amazon-linux-extras
# amazon-linux-extras install epel -y 
# yum-config-manager --enable epel
# yum install git-lfs
# git lfs install

# type -p yum-config-manager >/dev/null || sudo yum install yum-utils
# sudo yum-config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo
# sudo yum install gh

poetry install --with ipy --with dev