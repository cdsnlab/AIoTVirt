import microservices
from microservices.microservice import Microservice
import numpy as np
import cv2


def ms_from_str(service_name, class_args, parent_kwargs) -> Microservice:
    try:
        class_name = service_name.split("_")[0]
        class_ = getattr(microservices, class_name)
        parent_kwargs["name"] = service_name
        return class_(**class_args, parent_kwargs=parent_kwargs)
    except AttributeError:
        print("Class doesn't exist")


def resize_numpy(input_np, size: tuple):
    input_np = np.transpose(input_np, (1, 2, 0))
    a = cv2.resize(input_np, dsize=size)
    a = np.transpose(a, (2, 0, 1)).astype(np.float32) / 255
    return a
