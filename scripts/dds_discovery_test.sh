#!/usr/bin/env bash
# Cross-machine DDS discovery test: Pi <-> Jetson.
#
# Tests the thing that actually matters — does DATA FLOW — rather than whether
# `ros2 topic list` shows something. Listing failed while a direct `echo` with an
# explicit type succeeded, so listing is not a reliable signal.
#
# Four cases. The interesting axis is START ORDER:
#   A  Pi publishes first, Jetson subscribes later   (pre-existing publisher)
#   B  Jetson subscribes first, Pi publishes later   (new publisher)
#   C  Jetson publishes first, Pi subscribes later   (reverse of A)
#   D  Pi subscribes first, Jetson publishes later   (reverse of B)
#
# If A and C fail while B and D pass, discovery is start-order dependent:
# a node that comes up before its peers is permanently deaf.
#
# Usage:
#   ./dds_discovery_test.sh            # normal, both interfaces live
#   ./dds_discovery_test.sh --eth-only # pin DDS to the direct ethernet link
#
# --eth-only is the decisive experiment for the dual-homing hypothesis: if the
# failures disappear when both ends are pinned to 192.168.50.0/24, the cause is
# Fast DDS advertising locators on both WiFi and eth and peers picking wrong.

set +u
PI_IP=192.168.50.1
JETSON=Omni
SETTLE=${SETTLE:-15}      # how long a "pre-existing" endpoint runs before its peer starts
LISTEN=${LISTEN:-12}      # how long the subscriber waits for data
TOPIC=/dds_probe
TYPE=std_msgs/String

PI_SETUP="source ~/ros2_jazzy/install/setup.bash"
JETSON_SETUP="source /opt/ros/jazzy/setup.bash"

if [ "$1" = "--eth-only" ]; then
  echo "### Mode: DDS pinned to the direct ethernet link (192.168.50.0/24)"
  cat > /tmp/dds_eth_only.xml <<XML
<?xml version="1.0" encoding="UTF-8" ?>
<dds xmlns="http://www.eprosima.com">
  <profiles>
    <transport_descriptors>
      <transport_descriptor>
        <transport_id>eth_only</transport_id>
        <type>UDPv4</type>
        <interfaceWhiteList><address>PLACEHOLDER</address></interfaceWhiteList>
      </transport_descriptor>
    </transport_descriptors>
    <participant profile_name="p" is_default_profile="true">
      <rtps>
        <userTransports><transport_id>eth_only</transport_id></userTransports>
        <useBuiltinTransports>false</useBuiltinTransports>
      </rtps>
    </participant>
  </profiles>
</dds>
XML
  sed "s/PLACEHOLDER/192.168.50.1/" /tmp/dds_eth_only.xml > /tmp/dds_pi.xml
  sed "s/PLACEHOLDER/192.168.50.2/" /tmp/dds_eth_only.xml > /tmp/dds_jetson.xml
  scp -q /tmp/dds_jetson.xml $JETSON:/tmp/dds_jetson.xml
  PI_SETUP="$PI_SETUP; export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/dds_pi.xml"
  JETSON_SETUP="$JETSON_SETUP; export FASTRTPS_DEFAULT_PROFILES_FILE=/tmp/dds_jetson.xml"
else
  echo "### Mode: default (both WiFi and ethernet live)"
fi

cleanup() {
  pkill -f "topic pub $TOPIC"  2>/dev/null
  pkill -f "topic echo $TOPIC" 2>/dev/null
  ssh -o BatchMode=yes $JETSON 'pkill -f "topic pub /dds_probe"; pkill -f "topic echo /dds_probe"' 2>/dev/null
}
trap cleanup EXIT

pi_pub()     { bash -c "$PI_SETUP; exec ros2 topic pub -r 2 $TOPIC $TYPE '{data: probe}'" >/dev/null 2>&1 & }
pi_sub()     { bash -c "$PI_SETUP; exec timeout $LISTEN ros2 topic echo $TOPIC $TYPE" 2>/dev/null | head -2; }
jetson_pub() { ssh -o BatchMode=yes $JETSON "$JETSON_SETUP; nohup ros2 topic pub -r 2 $TOPIC $TYPE '{data: probe}' >/dev/null 2>&1 &" 2>/dev/null; }
jetson_sub() { ssh -o BatchMode=yes $JETSON "$JETSON_SETUP; timeout $LISTEN ros2 topic echo $TOPIC $TYPE 2>/dev/null | head -2" 2>/dev/null; }

report() { # $1=case name  $2=captured output
  if echo "$2" | grep -q probe; then echo "  RESULT: PASS — data received"
  else                               echo "  RESULT: FAIL — nothing received in ${LISTEN}s"; fi
}

echo
echo "=== A: Pi publishes FIRST (${SETTLE}s), then Jetson subscribes ==="
cleanup; sleep 2; pi_pub; sleep $SETTLE
out=$(jetson_sub); report A "$out"; cleanup; sleep 3

echo
echo "=== B: Jetson subscribes FIRST, then Pi publishes ==="
# NOTE: never `wait` here — the publisher is a background job that runs forever,
# so `wait` would block on it indefinitely. Wait on the subscriber PID only.
( jetson_sub > /tmp/dds_b.txt 2>&1 ) & sub_pid=$!
sleep 5; pi_pub; wait $sub_pid 2>/dev/null
report B "$(cat /tmp/dds_b.txt)"; cleanup; sleep 3

echo
echo "=== C: Jetson publishes FIRST (${SETTLE}s), then Pi subscribes ==="
jetson_pub; sleep $SETTLE
out=$(pi_sub); report C "$out"; cleanup; sleep 3

echo
echo "=== D: Pi subscribes FIRST, then Jetson publishes ==="
( pi_sub > /tmp/dds_d.txt 2>&1 ) & sub_pid=$!
sleep 5; jetson_pub; wait $sub_pid 2>/dev/null
report D "$(cat /tmp/dds_d.txt)"; cleanup

echo
echo "A/C FAIL + B/D PASS  => start-order dependent discovery (nodes that start early go deaf)"
echo "all PASS with --eth-only => dual-homing is the cause; pin DDS to eth0"
