#!/bin/bash
source /home/pi/ros2_jazzy/install/setup.bash
source /home/pi/omni_ws/install/setup.bash

# ament_python packages don't auto-register — mirror the .bashrc loop
for _pkg in /home/pi/omni_ws/install/*/; do
    if [ -d "${_pkg}share/ament_index/resource_index/packages" ]; then
        export AMENT_PREFIX_PATH="${_pkg%/}:$AMENT_PREFIX_PATH"
    fi
done

exec /home/pi/omni_ws/install/bms_node/lib/bms_node/bms_node
