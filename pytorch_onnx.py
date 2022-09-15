import os
import sys
import torch
import onnxruntime as ort

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.append(BASE_DIR)


def pytorch2onnx_convert(
    model: torch.nn.Module,
    onnx_folder_path: str,
    device='cuda',
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    },
    verbose=False,
    **model_specs
):
    if model_specs['task'] == 'classification':
        model_name = model_specs['name']
        batch_size = model_specs['batch_size']
        num_channels = model_specs['num_channels']
        height = model_specs['height']
        width = model_specs['width']
    model = model.eval().to(device)
    dummy_input = torch.randn(
        batch_size,
        num_channels,
        height,
        width,
        device=device
    )
    model_out_path = os.path.join(onnx_folder_path, model_name + '.onnx')
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=model_out_path,
        export_params=True,
        verbose=verbose,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return model_out_path


if __name__ == '__main__':
    device = ort.get_device()
    assert (device == 'GPU'), 'Device should be GPU, got {}'.format(device)
    pytorch2onnx_convert(
        os.path.join(BASE_DIR, 'distributed/cli/onnx_models'),
        os.path.join(BASE_DIR, 'distributed/cli/model_source_code'),
        task='classification',
        name='resnet18',
        batch_size=1,
        num_channels=3,
        height=224,
        width=224,
    )
