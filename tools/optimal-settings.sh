#!/bin/bash
#
# Increase memory buffers for GigE to 32MB
sudo sysctl -w net.core.rmem_max=33554432
sudo sysctl -w net.core.rmem_default=33554432
sudo sysctl -w net.core.wmem_max=33554432
sudo sysctl -w net.core.wmem_default=33554432
#
# Turn off USB autosuspend
sudo sh -c 'echo -1 > /sys/module/usbcore/parameters/autosuspend'
#
# Increase USB memory for support of > 2MB images
sudo sh -c 'echo 1000 > /sys/module/usbcore/parameters/usbfs_memory_mb'

