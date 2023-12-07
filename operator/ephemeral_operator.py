import kopf
import logging
import kubernetes as k8s
import os
import yaml

@kopf.on.create('ephemeralvolumeclaims')
def create_fn(spec, name, namespace, logger, **kwargs):

    size = spec.get('size')
    volume_name = spec.get('volume_name', None)

    if not size:
        raise kopf.PermanentError(f"Size must be set. Got {size!r}.")
    
    path = os.path.join(os.path.dirname(__file__), 'pv.yaml')
    tmpl = open(path, 'rt').read()
    text = tmpl.format(volume_name=volume_name,size=size)
    pv_data = yaml.safe_load(text)

    kopf.adopt(pv_data)

    api = k8s.client.CoreV1Api()
    obj = api.create_persistent_volume(
        body=pv_data,
    )

    logger.info(f"PV child is created: {obj}")

    path = os.path.join(os.path.dirname(__file__), 'pvc.yaml')
    tmpl = open(path, 'rt').read()
    text = tmpl.format(name=name, size=size, volume_name=volume_name)
    pvc_data = yaml.safe_load(text)

    kopf.adopt(pvc_data)

    api = k8s.client.CoreV1Api()
    obj = api.create_namespaced_persistent_volume_claim(
        namespace=namespace,
        body=pvc_data,
    )


    logger.info(f"PVC child is created: {obj}")

    return {'pvc-name': obj.metadata.name,'pv-name': obj.spec.volume_name}