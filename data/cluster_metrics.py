import kubernetes as k8s
import time 

k8s.config.load_kube_config()

#get pods and containers metrics
k8s_core_v1 = k8s.client.CoreV1Api()
api = k8s.client.CustomObjectsApi()
# pods = k8s_core_v1.list_namespaced_pod(namespace="default")
#{'kind': 'PodMetricsList', 'apiVersion': 'metrics.k8s.io/v1beta1', 'metadata': {}, 'items': 
#[{'metadata': {'name': 'dataread-deployment-7649bd9657-xpzdv', 'namespace': 'default', 'creationTimestamp': '2024-01-24T14:41:16Z', 'labels': {'app': 'data-reading-example', 'pod-template-hash': '7649bd9657'}}, 'timestamp': '2024-01-24T14:40:46Z', 'window': '17.215s', 'containers': [{'name': 'dataread', 'usage': {'cpu': '989681963n', 'memory': '1009716Ki'}}]}, 
# {'metadata': {'name': 'periodic-log-generator-job-gzqck', 'namespace': 'default', 'creationTimestamp': '2024-01-24T14:41:16Z', 'labels': {'batch.kubernetes.io/controller-uid': '09c31498-8702-4ae7-97c0-bbc5bb7e4510', 'batch.kubernetes.io/job-name': 'periodic-log-generator-job', 'controller-uid': '09c31498-8702-4ae7-97c0-bbc5bb7e4510', 'job-name': 'periodic-log-generator-job'}}, 'timestamp': '2024-01-24T14:40:51Z', 'window': '15.198s', 'containers': [{'name': 'sim-gen', 'usage': {'cpu': '19625213n', 'memory': '203844Ki'}}, {'name': 'logging-agent', 'usage': {'cpu': '991583778n', 'memory': '402216Ki'}}]}]}


with open("./data/cluster_metrics.txt","w") as f:
    f.write("node,pod,container,cpu,memory\n")


for i in range(100):
    resource = api.list_namespaced_custom_object(group="metrics.k8s.io",version="v1beta1", namespace="default", plural="pods")
    for pod in resource["items"]:
        pod_data = k8s_core_v1.read_namespaced_pod(name=pod["metadata"]["name"],namespace="default")
        node_name = pod_data.spec.node_name
        for container in pod["containers"]:
            with open("./data/cluster_metrics.txt","a") as f:
                f.write("{},{},{},{},{}\n".format(node_name,pod["metadata"]["name"],container["name"],container["usage"]["cpu"],container["usage"]["memory"]))
    time.sleep(1)

