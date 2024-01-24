import kubernetes as k8s

k8s.config.load_kube_config()

#get pods and containers metrics
k8s_core_v1 = k8s.client.CoreV1Api()
api = k8s.client.CustomObjectsApi()
resource = api.list_namespaced_custom_object(group="metrics.k8s.io",version="v1beta1", namespace="default", plural="pods")
# pods = k8s_core_v1.list_namespaced_pod(namespace="default")


for pod in resource["items"]:
    print(pod["metadata"]["name"])
    print("cpu usage: {}".format(pod["containers"][0]["usage"]["cpu"]))
    print("memory usage: {}".format(pod["containers"][0]["usage"]["memory"]))
    print("cpu limit: {}".format(pod["spec"]["containers"][0]["resources"]["limits"]["cpu"]))
    print("memory limit: {}".format(pod["spec"]["containers"][0]["resources"]["limits"]["memory"]))
    print("cpu request: {}".format(pod["spec"]["containers"][0]["resources"]["requests"]["cpu"]))
    print("memory request: {}".format(pod["spec"]["containers"][0]["resources"]["requests"]["memory"]))
    print("cpu usage: {}".format(pod["containers"][1]["usage"]["cpu"]))
    print("memory usage: {}".format(pod["containers"][1]["usage"]["memory"]))
    print("cpu limit: {}".format(pod["spec"]["containers"][1]["resources"]["limits"]["cpu"]))
    print("memory limit: {}".format(pod["spec"]["containers"][1]["resources"]["limits"]["memory"]))
    print("cpu request: {}".format(pod["spec"]["containers"][1]["resources"]["requests"]["cpu"]))
    print("memory request: {}".format(pod["spec"]["containers"][1]["resources"]["requests"]["memory"]))
    print("\n")
