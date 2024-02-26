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


@kopf.on.create('kopfexamples')
def create_fn(spec, **kwargs):

    # Render the pod yaml with some spec fields used in the template.
    doc = yaml.safe_load(f"""
        apiVersion: v1
        kind: Pod
        spec:
          containers:
          - name: the-only-one
            image: busybox
            command: ["sh", "-x", "-c"]
            args:
            - |
              echo "FIELD=$FIELD"
              sleep {spec.get('duration', 0)}
            env:
            - name: FIELD
              value: {spec.get('field', 'default-value')}
    """)

    # Make it our child: assign the namespace, name, labels, owner references, etc.
    kopf.adopt(doc)

    # Actually create an object by requesting the Kubernetes API.
    api = pykube.HTTPClient(pykube.KubeConfig.from_env())
    pod = pykube.Pod(api, doc)
    pod.create()
    api.session.close()

    # Update the parent's status.
    return {'children': [pod.metadata['uid']]}

@kopf.on.create('kopfexamples')
def print_fn(spec, **kwargs):
    print(f"Object is created with spec: {spec!r}")

@kopf.timer('kopfexamples', interval=5, initial_delay=5)
def timer_fn(**kwargs):
    print("Timer is fired.")
    return {'message': 'Hello from the timer!'}
