# Run Triton with FasterTransformer backend on GKE

```

# 1. Confgure environment variables
export PROJECT_ID=rthallam-demo-project
export ZONE=us-central1-a
export REGION=us-central1
export DEPLOYMENT_NAME=triton-gke

gcloud config set project $PROJECT_ID

# 2. Create GKE cluster
gcloud beta container clusters create ${DEPLOYMENT_NAME} \
--addons=HorizontalPodAutoscaling,HttpLoadBalancing,Istio \
--machine-type=n1-standard-4 \
--node-locations=${ZONE} \
--zone=${ZONE} \
--subnetwork=default \
--scopes cloud-platform \
--num-nodes 1 \
--project ${PROJECT_ID} # \
#--workload-pool=${PROJECT_ID}.svc.id.goog

# add workload identity
# gcloud container clusters update ${DEPLOYMENT_NAME} \
#     --zone=${ZONE} \
#     --workload-pool=${PROJECT_ID}.svc.id.goog
    
# gcloud container clusters update ${DEPLOYMENT_NAME} \
#     --zone=${ZONE} \
#     --disable-workload-identity

# 3. Add node pool with A100 to GKE cluster
gcloud beta container node-pools create accel-a100 \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --cluster ${DEPLOYMENT_NAME} \
  --num-nodes 1 \
  --accelerator type=nvidia-tesla-a100,count=1  \
  --enable-autoscaling --min-nodes 1 --max-nodes 1 \
  --machine-type=a2-highgpu-1g  \
  --disk-size=300 \
  --scopes cloud-platform \
  --verbosity error
#  \
#  --workload-metadata=GKE_METADATA

# 4. Get credentials to run kubectl
gcloud container clusters get-credentials ${DEPLOYMENT_NAME} --project ${PROJECT_ID} --zone ${ZONE}

# 5. <ake sure you can run kubectl locally to access the cluster
kubectl create clusterrolebinding cluster-admin-binding --clusterrole cluster-admin --user "$(gcloud config get-value account)"

# 6. Install NVIDIA device plugin for GKE to prepare GPU nodes for driver install
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml

# 7. Clone Triton repo
git clone https://github.com/triton-inference-server/server.git
cd server/deploy/gcp

# 8. Triton deploy config yaml
cat << EOF > ~/server/deploy/gcp/config.yaml
namespace: default
image:
  imageName: gcr.io/rthallam-demo-project/llms-on-vertex-ai/nemo-bignlp-triton-inference
  modelRepositoryPath: gs://cloud-ai-platform-2f444b6a-a742-444b-b91a-c7519f51bd77/llm/models/ft/
EOF

cat << EOF > ~/server/deploy/gcp/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ template "triton-inference-server.fullname" . }}
  namespace: {{ .Release.Namespace }}
  labels:
    app: {{ template "triton-inference-server.name" . }}
    chart: {{ template "triton-inference-server.chart" . }}
    release: {{ .Release.Name }}
    heritage: {{ .Release.Service }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ template "triton-inference-server.name" . }}
      release: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ template "triton-inference-server.name" . }}
        release: {{ .Release.Name }}

    spec:
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 5G

      containers:
        - name: {{ .Chart.Name }}
          image: "{{ .Values.image.imageName }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}

          resources:
            limits:
              nvidia.com/gpu: {{ .Values.image.numGpus }}

          args: ["tritonserver", "--model-repository={{ .Values.image.modelRepositoryPath }}", "--log-verbose=99", "--log-error=1"]

          ports:
            - containerPort: 8000
              name: http
            - containerPort: 8001
              name: grpc
            - containerPort: 8002
              name: metrics
          livenessProbe:
            failureThreshold: 60
            initialDelaySeconds: 10
            periodSeconds: 5
            successThreshold: 1
            timeoutSeconds: 1
            httpGet:
              path: /v2/health/live
              port: http
          readinessProbe:
            failureThreshold: 60
            initialDelaySeconds: 10
            periodSeconds: 5
            successThreshold: 1
            timeoutSeconds: 1
            httpGet:
              path: /v2/health/ready
              port: http

          volumeMounts:
            - mountPath: /dev/shm
              name: dshm
      securityContext:
        runAsUser: 1000
        fsGroup: 1000
EOF

# 9. Deploy Triton and check status
helm install example -f config.yaml .
kubectl get pod --selector="app=triton-inference-server"

# 10. Run health check
kubectl port-forward $(kubectl get pod --selector="app=triton-inference-server" \
  --output jsonpath='{.items[0].metadata.name}') 8000:8000
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/v2/health/ready


# ---------------------------------------------------------
# Add node pool with T4 GPU
gcloud container node-pools create accel-t4 \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --cluster ${DEPLOYMENT_NAME} \
  --num-nodes 1 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --enable-autoscaling --min-nodes 1 --max-nodes 2 \
  --machine-type n1-standard-4 \
  --disk-size=100 \
  --scopes cloud-platform \
  --verbosity error
#  \
#  --workload-metadata=GKE_METADATA

# Add node pool with A100 80GB disk
gcloud beta container node-pools create accel-a100-ultra \
  --project ${PROJECT_ID} \
  --zone ${ZONE} \
  --cluster ${DEPLOYMENT_NAME} \
  --num-nodes 1 \
  --accelerator type=nvidia-a100-80gb,count=1  \
  --enable-autoscaling --min-nodes 1 --max-nodes 1 \
  --machine-type=a2-ultragpu-1g  \
  --disk-size=100 \
  --scopes cloud-platform \
  --verbosity error

# Enable stackdriver custom metrics adaptor
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/k8s-stackdriver/master/custom-metrics-stackdriver-adapter/deploy/production/adapter.yaml

# Deploy Prometheus
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install example-metrics --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false prometheus-community/kube-prometheus-stack

# Mount prepopulated persistent disk on GKE cluster
cat <<EOF > preexisting-disk-pv-pvc.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: llm-workspace
spec:
  storageClassName: premium-rwo
  capacity:
    storage: 200G
  accessModes:
    - ReadOnlyMany
  claimRef:
    namespace: default
    name: llm-claim-workspace
  csi:
    driver: pd.csi.storage.gke.io
    volumeHandle: projects/rthallam-demo-project/zones/us-central1-a/disks/llm-ckpt
    fsType: ext4
    readOnly: true
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  namespace: default
  name: llm-claim-workspace
spec:
  storageClassName: premium-rwo
  volumeName: llm-workspace
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 200G
EOF
      
# mount disk on Triton config
          volumeMounts:
            - mountPath: /llm-ckpt
              name: llm-ckpt-volume
              readOnly: true
      volumes:
        - name: llm-ckpt-volume
          persistentVolumeClaim:
            claimName: llm-claim-workspace
            readOnly: true

# Apply persistent disk changes
kubectl apply -f ~/preexisting-disk-pv-pvc.yaml


# run sample test
curl -X POST -H "Content-Type: application/json" \
    --data-binary @payload.json \
    localhost:8000/v2/models/object_detector/infer | \
  jq -c '.outputs[] | select(.name == "detection_classes")'
  

# Cleanup
#   - Undeploy Triton deployment
# helm uninstall example
#   - Delete nodepool from cluster 
# gcloud container node-pools  delete  accel-t4   --project ${PROJECT_ID}   --zone ${ZONE}   --cluster ${DEPLOYMENT_NAME}
#   - Delete PV,PVC
kubectl delete persistentvolumeclaim llm-claim-workspace
kubectl delete persistentvolume llm-workspace

# Deploy NVIDIA device plugin for GKE to prepare GPU nodes for driver install, additional line to install MIG
# kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/cmd/nvidia_gpu/device-plugin.yaml
# kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-nvidia-mig.yaml
  
  
```