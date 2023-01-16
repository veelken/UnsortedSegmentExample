import torch
from torch import nn

class UnsortedSegmentExample_fixed(nn.Module):
    r''' Implimentation of LorentzNet.

    Args:
        - `mode`           (str)         : either "sum" to test unsorted_segment_sum or "mean" to test unsorted_segment_mean
        - `segment_ids`    (torch.Tensor): segment_ids parameter passed to unsorted_segment_sum and unsorted_segment_mean functions
        - `num_segments`   (torch.Tensor): num_segments parameter passed to unsorted_segment_sum and unsorted_segment_mean functions
    '''
    def __init__(self, mode : str, segment_ids : torch.Tensor, num_segments : torch.Tensor) -> None:
        print("<UnsortedSegmentExample_fixed::init>:")
        print(" mode = '%s'" % mode)
        super(UnsortedSegmentExample_fixed, self).__init__()

        self.mode = mode
        self.segment_ids = torch.nn.functional.one_hot(segment_ids, int(num_segments[0]))
        self.segment_ids = torch.transpose(self.segment_ids, 0, 1).float()

    def forward(self, data : torch.Tensor) -> torch.Tensor:
        if self.mode == "sum":
            return unsorted_segment_sum(data, self.segment_ids)
        elif self.mode == "mean":
            return unsorted_segment_mean(data, self.segment_ids)
        else:
            raise ValueError("Invalid parameter mode = '%s' !!" % self.mode)

def unsorted_segment_sum(data : torch.Tensor, segment_ids : torch.Tensor) -> torch.Tensor:
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    '''
    result = segment_ids @ data
    return result

def unsorted_segment_mean(data : torch.Tensor, segment_ids : torch.Tensor) -> torch.Tensor:
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    '''
    result = segment_ids @ data
    count = segment_ids @ torch.ones_like(data)
    result = result / count.clamp(min=1)
    return result
