import kubernetes as k8s
import dns
import dns.resolver
import dns.rdataclass
import dns.rdatatype
import socket


k8s.config.load_kube_config()

#get endpoints
endpoints = k8s.client.CoreV1Api().list_namespaced_endpoints("default").items
#filter reader-service endpoint
endpoints = [e for e in endpoints if e.metadata.name == "reader-service"]
print(endpoints)
#get ip and port
ips = [s.ip for s in endpoints[0].subsets[0].addresses]
port = endpoints[0].subsets[0].ports[0].port

ips_port = [f"{ip}:{port}" for ip in ips]

print(ips_port)

endpoints = dns.resolver.resolve("reader-service.default.svc.cluster.local", "A")
print(endpoints.rrset)