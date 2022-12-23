#!/usr/bin/python3

import numpy as np
import torch
from torch import nn
import torch.onnx
import onnx
import onnxruntime
import argparse
from UnsortedSegmentExample import UnsortedSegmentExample
from torch.nn.parallel import DataParallel
import platform

parser = argparse.ArgumentParser(description='test conversion of unsorted_segment_sum and unsorted_segment_mean function from PyTorch to ONNX')
parser.add_argument('--mode', type=str, default='', metavar='N',
                    help='either "sum" to test unsorted_segment_sum or "mean" to test unsorted_segment_mean')
parser.add_argument('--outputFile', type=str, default='', metavar='N',
                    help='name of file to which ONNX model is written')

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == "__main__":
    ### initialize args
    args = parser.parse_args()

    ### print version information
    print("Printing version information...")
    print(" python = %s" % platform.python_version())
    print(" torch = %s" % torch.__version__)
    print(" onnx = %s" % onnx.__version__)
    print(" onnxruntime = %s" % onnxruntime.__version__)
    print(" ...Done.")

    ### initialize cuda
    print("Initializing CUDA...")
    device = torch.device("cpu")
    print(" ...Done.")

    ### define test data
    ### (list of particle four-vectors in the format [ Energy, Py, Py, Pz ])
    print("Creating test data...")
    raw_data = [
        [ 5.6791, -3.0550,  3.9282,  2.7364],
        [ 5.6741, -2.0466,  2.7770,  4.5028],
        [ 1.9659, -1.0984,  1.3296,  0.9436],
        [ 2.6953, -0.9157,  1.4559,  2.0705],
        [ 1.4291, -0.8965,  0.7097,  0.8573],
        [ 1.1897, -0.4098,  0.8424,  0.7200],
        [ 1.0954, -0.6871,  0.4854,  0.7015],
        [ 0.9791, -0.4832,  0.5853,  0.6185],
        [ 1.0496, -0.3403,  0.6656,  0.7234],
        [ 0.9958, -0.2716,  0.5650,  0.7737],
        [ 0.8529, -0.3361,  0.5252,  0.5819],
        [ 0.7956, -0.3299,  0.5177,  0.5061],
        [ 0.6419, -0.2421,  0.4817,  0.3484],
        [ 0.5590, -0.3271,  0.3542,  0.2828],
        [ 0.5984, -0.3497,  0.2972,  0.3841],
        [ 0.5018, -0.1784,  0.4062,  0.2344],
        [ 0.4682, -0.2234,  0.3363,  0.2370],
        [ 0.4826, -0.2560,  0.2663,  0.2775]
    ]
    data = torch.Tensor(raw_data).to(device, torch.float32)
    print("data = ", data)
    raw_segment_ids = [ 
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5 
    ]
    assert len(raw_data) == len(raw_segment_ids)
    segment_ids = torch.Tensor(raw_segment_ids).to(device, torch.int64)
    print("segment_ids = ", segment_ids) 
    num_segments = 6
    print(" ...Done.")

    ### compute expected output
    print("Computing expected output...")
    raw_expected_output = []
    for segment_id in range(num_segments):
        sum   = [ 0., 0., 0., 0. ]
        count = 0
        for idx in range(len(raw_data)):
            if raw_segment_ids[idx] == segment_id:
                assert len(raw_data[idx]) == 4
                for component in range(len(raw_data[idx])):
                    sum[component] += raw_data[idx][component]
                count += 1
        if args.mode == "sum":
            raw_expected_output.append(sum)
        elif args.mode == "mean":
            raw_expected_output.append(sum / count)
        else:
            raise ValueError("Invalid parameter mode = '%s' !!" % self.mode)
    expected_output = torch.Tensor(raw_expected_output).to(device, torch.float32)
    print("expected_output = ", expected_output) 
    print(" ...Done.")

    ### create PyTorch model
    print("Creating PyTorch model...")
    torch_model = UnsortedSegmentExample(args.mode, segment_ids, num_segments)
    print(" ...Done.")

    ### switch PyTorch model to inference mode
    print("Switching PyTorch model to inference mode...")
    torch_model.eval()
    print(" ...Done.")

    ### compute output of PyTorch model
    print("Computing output of PyTorch model...")
    torch_out = torch_model(data)
    print("torch_out = ", torch_out) 
    print(" ...Done.")

    ### check that ONNX model computes the same output as PyTorch model
    print("Checking that PyTorch model has computed the expected output...")
    np.testing.assert_allclose(expected_output, to_numpy(torch_out), rtol=1e-03, atol=1e-05)
    print(" ...Done.")

    ### generate object of type 'ScriptModule' for PyTorch model
    print("Converting PyTorch model to ONNX format...")
    torch_script = torch.jit.script(torch_model)
    print(" ...Done.")

    ### try to 'freeze' PyTorch model 
    ### to check for error reported at https://github.com/pytorch/pytorch/issues/71548
    print("Trying to 'freeze' PyTorch model...")
    torch.jit.freeze(torch_script)
    print(" ...Done.")

    ### save ONNX model
    ### NB.: syntax for calling 'export' function based on example taken from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html,
    ###      except for the 'dynamic_axes' parameter, which is taken from https://deci.ai/blog/how-to-convert-a-pytorch-model-to-onnx/
    print("Saving ONNX model to file '%s'..." % args.outputFile)
    dynamic_axes_dict = {
      'data' : {
        0 : 'n_particles'
      }
    }  
    torch.onnx.export(torch_script,              # model being run
                      (data),                    # model input (or a tuple for multiple inputs)
                      args.outputFile,           # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      opset_version=13,          # the ONNX version to export the model to
                      do_constant_folding=False, # whether to execute constant folding for optimization
                      input_names = ['data'],    # the model's input names 
                      output_names = ['out'],    # the model's output names
                      dynamic_axes = dynamic_axes_dict)
    print(" ...Done.")

    ### check that ONNX model has valid schema
    print("Checking that ONNX model contained in  file '%s' has valid schema..." % args.outputFile)
    onnx_model = onnx.load(args.outputFile)
    onnx.checker.check_model(onnx_model)
    print(" ...Done.")

    ### check that ONNX model computes the same output as PyTorch model
    print("Checking that ONNX model computes the same output as PyTorch model...")
    onnx_session = onnxruntime.InferenceSession(args.outputFile)
    onnx_inputs = {
      'data' : to_numpy(data)
    }
    onnx_outs = onnx_session.run(None, onnx_inputs) 
    print("onnx_out = ", onnx_outs[0])
    np.testing.assert_allclose(to_numpy(torch_out), onnx_outs[0], rtol=1e-03, atol=1e-05)
    print(" ...Done.")

    ### Done. All checks have passed :)
    print("--> Exported model has been tested with ONNXRuntime, and the result looks good.")
