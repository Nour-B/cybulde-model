#!/bin/bash

gcloud compute instances create-with-container etcd-server \
    --project=cybulde-n \
    --zone=europe-west4-b \
    --machine-type=n1-standard-1 \
    --network-interface=stack-type=IPV4_ONLY,subnet=default,no-address \
    --maintenance-policy=MIGRATE \
    --provisioning-model=STANDARD \
    --service-account=742391789554-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/trace.append \
    --image=projects/cos-cloud/global/images/cos-stable-117-18613-75-7 \
    --boot-disk-size=10GB \
    --boot-disk-type=pd-balanced \
    --boot-disk-device-name=etcd-server \
    --container-image=docker.io/bitnami/etcd:3.5 \
    --container-restart-policy=always \
    --container-privileged \
    --container-env=ALLOW_NONE_AUTHENTICATION=yes,ETCD_ADVERTISE_CLIENT_URLS=http://0.0.0.0:2379,ETCD_ENABLE_V2=true,ETCDCTL_API=2 \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud,container-vm=cos-stable-117-18613-75-7