# This is the project requirements

# AIDE1
a. Pytest CI/CD with 3 steps test, build, deploy.
b. Metrics tracking system + system monitoring dashboard (Prometheus + Grafana or relavant tools).
c. Evidently or relavant tools with data drift dashboard.
d. Tracing system (Jaeger, Loki or relavant tools).
e. Logging system (ELK, Loki or relavant tools).
f. Pre/Post-processing API with FastAPI (autoscale with HPA).
g. K-Serve for model serving, can scale down to 0.
h. NGINX API Gateway with authentication.
i. IaC (Infrastructure as Code) to provisioning Kubernetes.
j. Using MLFLow and DVC to version model and data, model will pull from MLFLow to build image, and deploy.
## Note:
k. Mandatory using Python
i. Mandatory using Helm to deploy K8S applications, can research Helmfile to monitor many Helm chart as a same time.
m. Mandatory using Kubernetes on Cloud.
n. Automated CI/CD Pipeline from stage test to build if test coverage > 80%, and stage build to deploy is manual trigger. This means test coverage over 80% is mandatory.

# AIDE4
a.