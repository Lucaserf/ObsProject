#using tc set a delay in the network

tc qdisc add dev eth0 root netem delay 400ms


#restore the network to normal

tc qdisc del dev eth0 root netem delay 400ms


#list all calico devices from ip a, and get the device name

ip a | grep cali | awk '{print $2}' | cut -d: -f1

# set a delay in all calico devices

for i in $(ip a | grep cali | awk '{print $2}' | cut -d: -f1); do tc qdisc add dev $i root netem delay 400ms; done

for i in $(ip a | grep cali | awk '{print $2}' | cut -d: -f1); do tc qdisc del dev $i root netem delay 400ms; done
