# UnsortedSegmentExample
Demonstrate difference in output between ONNX model created via torch.onnx.export and original PyTorch model

The issue has been submitted to the PyTorch developers forum,
at the address:
  https://github.com/pytorch/pytorch/issues/91357

# To Install

git clone https://github.com/veelken/UnsortedSegmentExample

git remote set-url origin git+ssh://git@github.com/veelken/UnsortedSegmentExample.git

# To Run

cd UnsortedSegmentExample

python3 test_UnsortedSegmentExample.py --mode="sum" -outputFile="test_UnsortedSegmentExample.onnx"

