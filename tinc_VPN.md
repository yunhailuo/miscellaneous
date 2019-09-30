# Install tinc VPN on Ubuntu 18.04 LTS

## Install dependencies
* tinc VPN needs LibreSSL or OpenSSL, zlib and lzo libraries with both the development AND runtime versions.

  `sudo apt-get install libssl libssl-dev liblzo2 liblzo2-dev zlib1g zlib1g-dev net-tools dnsutils`

* "net-tools" is required for changing route table with `route` thus only required by clients
* "dnsutils" is required for finding IP address through command line thus only required by server with cloud-init
* Some packages may not have the exact name. Use following commands or similar commands to find the right package:

  `apt list | grep libssl`

## Install tinc
* Install one of the precompiled packages.

  `sudo apt-get install tinc`

## Configuration
Using a "netname" for tinc VPN is required when running more than one tinc daemon on one computer and is highly recommended when running only one tinc daemon. Using the same "netname" for both server and client doesn't seem to be required but can be easy for management. In general, this "netname" is set by the directory name a user created for tinc configurations. It is then used by `tincd`-n option and used as the VPN interface name. Configurations below use "tincvpn0" for netname.

1. __Configure a server__

  * Create the configuration directory structure: `sudo mkdir -p /etc/tinc/tincvpn0/hosts`

  * Setup tinc main configuration (`sudo vi /etc/tinc/tincvpn0/tinc.conf`) with the following variables. Ubuntu 18.04 LTS should have necessary device files by default.

    ```
    Name = tincserver1
    Device = /dev/net/tun
    AddressFamily = ipv4
    ```

  * Setup tinc host configuration (`sudo vi /etc/tinc/tincvpn0/hosts/tincserver1`) with the following variables and server's real public IP address.

    ```
    Address = 1.2.3.4
    Subnet = 192.168.10.0/24
    Subnet = 0.0.0.0/0

    ```

  * Uncomment "#net.ipv4.ip_forward=1" in "/etc/sysctl.conf" to allow forwarding of packets on server

    `sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf`

2. __Configure a client__

  * Create the configuration directory structure: `sudo mkdir -p /etc/tinc/tincvpn0/hosts`

  * Setup tinc main configuration (`sudo vi /etc/tinc/tincvpn0/tinc.conf`) with the following variables. Ubuntu 18.04 LTS should have necessary device files by default.

    ```
    Name = tincclient1
    Device = /dev/net/tun
    AddressFamily = ipv4
    ConnectTo = tincserver1
    ```

  * Setup tinc host configuration (`sudo vi /etc/tinc/tincvpn0/hosts/tincclient1`) with `Subnet = 192.168.10.2`.

## Generate keypairs

* The following commands by default will create the private key (e.g. /etc/tinc/tincserver1/rsa_key.priv) and appends the public key to the host configuration file

  `sudo tincd -n tincvpn0 -K4096`

* Make private keys root and read only

  `sudo chmod 400 /etc/tinc/tincvpn0/rsa_key.priv`

## Setup start up and tear down scripts

1. Scripts on the server:

  * The start up script (`sudo vi /etc/tinc/tincvpn0/tinc-up`) on the server will setup a network interface for tinc VPN and open port 655 for tinc. Moreover, it will setup port forwarding and IP Masquerade to relay packets between tinc network (192.168.10.0/24) on "tincvpn0" interface and server's public network on "ens3" interface. Make sure port 655 is open by other policies/rules.

    ```
    #!/bin/sh
    ifconfig $INTERFACE 192.168.10.1 netmask 255.255.255.0

    iptables -I INPUT -p tcp --dport 655 -j ACCEPT
    iptables -I FORWARD -s 192.168.10.0/24 -i tincvpn0 -o ens3 -j ACCEPT
    iptables -I FORWARD -d 192.168.10.0/24 -i ens3 -o tincvpn0 -j ACCEPT
    iptables -t nat -I POSTROUTING -s 192.168.10.0/24 -o ens3 -j MASQUERADE
    ```

  * The tear down script (`sudo vi /etc/tinc/tincvpn0/tinc-down`) will remove the "tincvpn0" interface, close port 655, stop port forwarding and stop IP Masquerade.

    ```
    #!/bin/sh
    ifconfig $INTERFACE down

    iptables -D INPUT -p tcp --dport 655 -j ACCEPT
    iptables -D FORWARD -s 192.168.10.0/24 -i tincvpn0 -o ens3 -j ACCEPT
    iptables -D FORWARD -d 192.168.10.0/24 -i ens3 -o tincvpn0 -j ACCEPT
    iptables -t nat -D POSTROUTING -s 192.168.10.0/24 -o ens3 -j MASQUERADE
    ```

2. Scripts on a client:

  * The start up script (`sudo vi /etc/tinc/tincvpn0/tinc-up`) assumes port 655 is open on the client and will only setup a network interface for tinc VPN.

    ```
    #!/bin/sh
    ifconfig $INTERFACE 192.168.10.2 netmask 255.255.255.0
    ```

  * The tear down script (`sudo vi /etc/tinc/tincvpn0/tinc-down`) will remove the "tincvpn0" interface.

    ```
    #!/bin/sh
    ifconfig $INTERFACE down
    ```

  * When the server becomes reachable, the server start up script on the client (`sudo vi /etc/tinc/tincvpn0/hosts/tincserver1-up`) will change client's route table to forward all of client network traffic to the server:

    ```
    #!/bin/sh
    ORIGINAL_GATEWAY=`ip route show | grep ^default | cut -d ' ' -f 2-5`

    ip route add $REMOTEADDRESS $ORIGINAL_GATEWAY
    ip route add 0.0.0.0/1 dev $INTERFACE
    ip route add 128.0.0.0/1 dev $INTERFACE
    ```

  * When the server goes down, the server tear down script on the client (`sudo vi /etc/tinc/tincvpn0/hosts/tincserver1-down`) will reset client's route table:

    ```
    #!/bin/sh
    ORIGINAL_GATEWAY=`ip route show | grep ^default | cut -d ' ' -f 2-5`

    ip route del $REMOTEADDRESS $ORIGINAL_GATEWAY
    ip route del 0.0.0.0/1 dev $INTERFACE
    ip route del 128.0.0.0/1 dev $INTERFACE
    ```

3. Make all scripts executable

  `sudo chmod 755 /etc/tinc/tincvpn0/tinc-*`

  `sudo chmod 755 /etc/tinc/tincvpn0/hosts/tincserver1-*`

## Exchange keypairs (host files)
Copy host files so that every tinc nodes (server and client) have the same set of files under "/etc/tinc/tincvpn0/hosts/"

## Test tinc
1. Use the following command to start tinc daemons on all nodes (server and client):

  `sudo tincd -n tincvpn0 -D -d5`

2. Ping each other on all tinc nodes

  * Server: `ping 192.168.10.2`
  * Client: `ping 192.168.10.1`

3. Ping a public IP (8.8.8.8, for example) and check for public IP address on client

## Enable tinc service

`sudo systemctl enable tinc.service`

## Troubleshoot notes
* Test port accessibility from localhost:

  `netcat -v localhost 655`

* List ports being listened:

  `netstat -tulnp | grep LISTEN`

* Check which application is using a port:

  `sudo lsof -i tcp:655`

* Check firewall policies:

  `sudo iptables -L -v --line-numbers`

  `sudo iptables -t nat -L -v --line-numbers`

* Check route table:

  `route -n`

## Example cloud-init script for Ubuntu 18.04 server on cloud

```
#cloud-config
# vim: syntax=yaml

write_files:
-   content: |
        Name = tincserver1
        Device = /dev/net/tun
        AddressFamily = ipv4
    owner: root:root
    path: /etc/tinc/tincvpn0/tinc.conf
    permissions: '0644'
-   content: |
        Address = 1.2.3.4
        Subnet = 192.168.10.0/24
        Subnet = 0.0.0.0/0

    owner: root:root
    path: /etc/tinc/tincvpn0/hosts/tincserver1
    permissions: '0644'
-   content: |
        #!/bin/sh
        ifconfig $INTERFACE 192.168.10.1 netmask 255.255.255.0

        iptables -I INPUT -p tcp --dport 655 -j ACCEPT
        iptables -I FORWARD -s 192.168.10.0/24 -i tincvpn0 -o ens3 -j ACCEPT
        iptables -I FORWARD -d 192.168.10.0/24 -i ens3 -o tincvpn0 -j ACCEPT
        iptables -t nat -I POSTROUTING -s 192.168.10.0/24 -o ens3 -j MASQUERADE
    owner: root:root
    path: /etc/tinc/tincvpn0/tinc-up
    permissions: '0644'
-   content: |
        #!/bin/sh
        ifconfig $INTERFACE down

        iptables -D INPUT -p tcp --dport 655 -j ACCEPT
        iptables -D FORWARD -s 192.168.10.0/24 -i tincvpn0 -o ens3 -j ACCEPT
        iptables -D FORWARD -d 192.168.10.0/24 -i ens3 -o tincvpn0 -j ACCEPT
        iptables -t nat -D POSTROUTING -s 192.168.10.0/24 -o ens3 -j MASQUERADE
    owner: root:root
    path: /etc/tinc/tincvpn0/tinc-down
    permissions: '0644'

packages:
  - libssl1.1
  - libssl-dev
  - liblzo2-2
  - liblzo2-dev
  - zlib1g
  - zlib1g-dev
  - net-tools
  - tinc
  - dnsutils
package_update: true
package_upgrade: true

runcmd:
  - sudo apt-get --yes autoremove
  - sudo mkdir -p /etc/tinc/tincvpn0/hosts
  - sudo tincd -n tincvpn0 -K4096
  - sudo chmod 400 /etc/tinc/tincvpn0/rsa_key.priv
  - sudo chmod 755 /etc/tinc/tincvpn0/tinc-*
  - sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf
  - sudo sed -i "s/1.2.3.4/$(dig +short myip.opendns.com @resolver1.opendns.com)/" /etc/tinc/tincvpn0/hosts/tincserver1
  - sudo systemctl enable tinc.service
  - sudo systemctl enable tinc@tincvpn0.service
  - sudo reboot
```

## Example script for setting client on Ubuntu 18.04

```
#!/bin/bash

sudo apt-get --yes install libssl1.1 libssl-dev liblzo2-2 liblzo2-dev zlib1g zlib1g-dev net-tools
sudo apt-get install tinc
sudo mkdir -p /etc/tinc/tincvpn0/hosts
sudo tee -a /etc/tinc/tincvpn0/tinc.conf << EOF
Name = tincclient1
Device = /dev/net/tun
AddressFamily = ipv4
ConnectTo = tincserver1
EOF
echo "Subnet = 192.168.10.2" | sudo tee -a /etc/tinc/tincvpn0/hosts/tincclient1
sudo tincd -n tincvpn0 -K4096
sudo chmod 400 /etc/tinc/tincvpn0/rsa_key.priv
sudo tee -a /etc/tinc/tincvpn0/tinc-up << EOF
#!/bin/sh
ifconfig \$INTERFACE 192.168.10.2 netmask 255.255.255.0
EOF
sudo tee -a /etc/tinc/tincvpn0/tinc-down << EOF
#!/bin/sh
ifconfig \$INTERFACE down
EOF
sudo tee -a /etc/tinc/tincvpn0/hosts/tincserver1-up << EOF
#!/bin/sh
ORIGINAL_GATEWAY=\`ip route show | grep ^default | cut -d ' ' -f 2-5\`

ip route add \$REMOTEADDRESS \$ORIGINAL_GATEWAY
ip route add 0.0.0.0/1 dev \$INTERFACE
ip route add 128.0.0.0/1 dev \$INTERFACE
EOF
sudo tee -a /etc/tinc/tincvpn0/hosts/tincserver1-down << EOF
#!/bin/sh
ORIGINAL_GATEWAY=\`ip route show | grep ^default | cut -d ' ' -f 2-5\`

ip route del \$REMOTEADDRESS \$ORIGINAL_GATEWAY
ip route del 0.0.0.0/1 dev \$INTERFACE
ip route del 128.0.0.0/1 dev \$INTERFACE
EOF
sudo chmod 755 /etc/tinc/tincvpn0/tinc-*
sudo chmod 755 /etc/tinc/tincvpn0/hosts/tincserver1-*

# Exchange keypairs and test tinc
# sudo tincd -n tincvpn0 -D -d5
# ping 192.168.10.2
# ping 192.168.10.1
# ping 8.8.8.8
```
