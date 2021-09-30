from kubernetes import client, config
from kubernetes.client.rest import ApiException
import datetime, time


# Taken from https://github.com/kubernetes-client/python/issues/571
def _wait_for_restart_complete(read, name, namespace, timeout=300, progress=None, restart_status=None):
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(5)
        response = read(name, namespace)
        s = response.status
        updated = s.updated_number_scheduled if s.updated_number_scheduled else 0
        if (updated >= s.desired_number_scheduled):
            progress.progress(0)
            return True
        else:
            print("Waiting for rollout to finish: {} out of {} new pods have been updated...".format(updated, s.desired_number_scheduled))
            restart_status.write("{} out of {} pods have been updated".format(updated, s.desired_number_scheduled))
            progress.progress(int(updated / s.desired_number_scheduled * 100))

    raise RuntimeError(f'Waiting timeout for deployment {name}')

# Taken from https://stackoverflow.com/a/67491253
def _restart(patch, name, namespace):
    now = datetime.datetime.utcnow()
    now = str(now.isoformat("T") + "Z")
    body = {
        'spec': {
            'template':{
                'metadata': {
                    'annotations': {
                        'kubectl.kubernetes.io/restartedAt': now
                    }
                }
            }
        }
    }
    try:
        patch(name, namespace, body, pretty='true')
    except ApiException as e:
        print("Exception when calling AppsV1Api->read_namespaced_deployment_status: %s\n" % e)

def restart_application(type, name, namespace, progress = None, restart_status = None, restart = True):
    config.load_kube_config("kubeconfig")
    v1_apps = client.AppsV1Api()
    patch = None
    read = None
    if type == "deployment":
        patch = v1_apps.patch_namespaced_deployment
        read  = v1_apps.read_namespaced_deployment_status
    elif type == "daemonset":
        patch = v1_apps.patch_namespaced_daemon_set
        read  = v1_apps.read_namespaced_daemon_set_status       

    if restart:
        _restart(patch, name, namespace)
    return _wait_for_restart_complete(read, name, namespace, 
                                            progress = progress, restart_status = restart_status)
