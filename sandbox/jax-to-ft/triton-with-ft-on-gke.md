# Run Triton with FasterTransformer backend on GKE

```
export PROJECT_ID=rthallam-demo-project
export ZONE=us-central1-a
export REGION=us-central1
export DEPLOYMENT_NAME=triton-gke

# create GKE cluster
gcloud beta container clusters create ${DEPLOYMENT_NAME} \
--addons=HorizontalPodAutoscaling,HttpLoadBalancing,Istio \
--machine-type=n1-standard-8 \
--node-locations=${ZONE} \
--zone=${ZONE} \
--subnetwork=default \
--scopes cloud-platform \
--num-nodes 1 \
--project ${PROJECT_ID}

# add node pool
gcloud beta container node-pools create accel \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --cluster ${DEPLOYMENT_NAME} \
  --num-nodes 1 \
  --accelerator type=nvidia-tesla-a100,count=1  \
  --enable-autoscaling --min-nodes 1 --max-nodes 2 \
  --machine-type=a2-highgpu-1g  \
  --disk-size=100 \
  --scopes cloud-platform \
  --verbosity error

# get credentials
gcloud container clusters get-credentials ${DEPLOYMENT_NAME} --project ${PROJECT_ID} --zone ${ZONE}

# deploy NVIDIA device plugin for GKE to prepare GPU nodes for driver install, additional line to install MIG
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/cmd/nvidia_gpu/device-plugin.yaml
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-nvidia-mig.yaml

# deploy Triton Inference server using NVIDIA bignlp container with FT dependencies
# https://github.com/triton-inference-server/server/tree/main/deploy/gke-marketplace-app
# https://github.com/triton-inference-server/server/tree/main/deploy/gcp
cd <directory containing Chart.yaml>
helm install example .
```