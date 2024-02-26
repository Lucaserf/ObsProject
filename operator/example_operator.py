# import kopf
# import logging
# import kubernetes as k8s
# import os
# import yaml
# import asyncio

# @kopf.on.create('ephemeralvolumeclaims')
# def create_fn(spec, name, namespace, logger, **kwargs):
#     logging.info(f"A handler is called for creating a resource {name!r} in namespace {namespace!r}.")

#     size = spec.get('size')
#     volume_name = spec.get('volume_name', None)

#     if not size:
#         raise kopf.PermanentError(f"Size must be set. Got {size!r}.")
    
#     path = os.path.join(os.path.dirname(__file__), 'pv.yaml')
#     tmpl = open(path, 'rt').read()
#     text = tmpl.format(volume_name=volume_name,size=size)
#     pv_data = yaml.safe_load(text)

#     kopf.adopt(pv_data)

#     api = k8s.client.CoreV1Api()
#     obj = api.create_persistent_volume(
#         body=pv_data,
#     )

#     logger.info(f"PV child is created: {obj}")

#     path = os.path.join(os.path.dirname(__file__), 'pvc.yaml')
#     tmpl = open(path, 'rt').read()
#     text = tmpl.format(name=name, size=size, volume_name=volume_name)
#     pvc_data = yaml.safe_load(text)

#     kopf.adopt(pvc_data)

#     api = k8s.client.CoreV1Api()
#     obj = api.create_namespaced_persistent_volume_claim(
#         namespace=namespace,
#         body=pvc_data,
#     )


#     logger.info(f"PVC child is created: {obj}")

#     return {'pvc-name': obj.metadata.name,'pv-name': obj.spec.volume_name}


# # @kopf.daemon('ephemeralvolumeclaims')
# # async def daemon_fn(**kwargs):
# #     while True:
# #         logger = logging.getLogger(__name__)
# #         logger.info("Daemon is running")
# #         await asyncio.sleep(1)

import kopf
import pykube
import yaml
import time

max_memory = 64
max_cpu = 250

@kopf.on.create('kopfexamples')
def create_fn(spec,**kwargs):
  api = pykube.HTTPClient(pykube.KubeConfig.from_env())

  pods = pykube.Pod.objects(api).filter(namespace="default", selector={'child': 'kopfexample'})
  state = len(pods.response['items'])


  if state < 1:
      text_to_print = "I am the only one"
      memory = max_memory
      cpu = max_cpu
  else:
      text_to_print = "I am not alone"
      memory = max_memory//(state+1)
      cpu = max_cpu//(state+1)

  # Render the pod yaml with some spec fields used in the template.
  doc = yaml.safe_load(f"""
      apiVersion: v1
      kind: Pod
      metadata:
        name: {spec.get('name', 'default-name')}
        labels: 
          child: 'kopfexample'
      spec:
        containers:
        - name: the-only-one
          image: busybox
          command: ["sh", "-x", "-c"]
          args:
          - |
            while true
            do
            echo "{text_to_print}, memory: {memory}Mi, cpu: {cpu}m"
            sleep 10
            done
          resources:
            limits:
              memory: "{memory}Mi"
              cpu: "{cpu}m"
  """)

  # Make it our child: assign the namespace, name, labels, owner references, etc.
  kopf.adopt(doc)

  # Actually create an object by requesting the Kubernetes API.
  pod = pykube.Pod(api, doc)
  pod.create()
  api.session.close()

  # Update the parent's status.
  return {"message": f"I am pod number {state}"}

@kopf.timer('kopfexamples', interval=10, initial_delay=10)
def timer_fn(spec,**kwargs):
  print(spec)
  api = pykube.HTTPClient(pykube.KubeConfig.from_env())
  state = 0
  
  pods = pykube.Pod.objects(api).filter(namespace="default", selector={'child': 'kopfexample'})
  state = len(pods.response['items'])

  if state <= 1:
      text_to_print = "I am the only one"
      memory = max_memory
      cpu = max_cpu
  else:
      text_to_print = "I am not alone"
      memory = max_memory//(state)
      cpu = max_cpu//(state)

  for pod in pods:
    name = pod.obj['metadata']['name']
    memory_assigned = pod.obj['spec']['containers'][0]['resources']['limits']['memory']
    print(name)
    if name == spec.get('name', 'default-name') and memory_assigned != f"{memory}Mi":
      pod.delete()
      #Update the pod memory and cpu
      doc = yaml.safe_load(f"""
      apiVersion: v1
      kind: Pod
      metadata:
        name: {spec.get('name', 'default-name')}
        labels: 
          child: 'kopfexample'
      spec:
        containers:
        - name: the-only-one
          image: busybox
          command: ["sh", "-x", "-c"]
          args:
          - |
            while true
            do
            echo "{text_to_print}, memory: {memory}Mi, cpu: {cpu}m"
            sleep 10
            done
          resources:
            limits:
              memory: "{memory}Mi"
              cpu: "{cpu}m"
      """)

      print(f"updating {name}, {memory}Mi, {cpu}m")
      kopf.adopt(doc)
      pods_names = [p['metadata']['name'] for p in pods.response['items']]
      while name in pods_names:
        time.sleep(1)
        pods = pykube.Pod.objects(api).filter(namespace="default", selector={'child': 'kopfexample'})
        pods_names = [p['metadata']['name'] for p in pods.response['items']]
      pod = pykube.Pod(api, doc)
      pod.create()


  api.session.close()
  return {'message': f'there are {state} custom pods running'}
