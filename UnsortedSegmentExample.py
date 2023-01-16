import torch
from torch import nn

class UnsortedSegmentExample(nn.Module):
    r''' Implimentation of LorentzNet.

    Args:
        - `mode`           (str)         : either "sum" to test unsorted_segment_sum or "mean" to test unsorted_segment_mean
        - `segment_ids`    (torch.Tensor): segment_ids parameter passed to unsorted_segment_sum and unsorted_segment_mean functions
        - `num_segments`   (torch.Tensor): num_segments parameter passed to unsorted_segment_sum and unsorted_segment_mean functions
    '''
    def __init__(self, mode : str, segment_ids : torch.Tensor, num_segments : torch.Tensor) -> None:
        print("<UnsortedSegmentExample::init>:")
        print(" mode = '%s'" % mode)
        super(UnsortedSegmentExample, self).__init__()

        self.mode = mode
        self.segment_ids = segment_ids
        self.num_segments = int(num_segments[0])

    def forward(self, data : torch.Tensor) -> torch.Tensor:
        if self.mode == "sum":
            return unsorted_segment_sum(data, self.segment_ids, self.num_segments)
        elif self.mode == "mean":
            return unsorted_segment_mean(data, self.segment_ids, self.num_segments)
        else:
            raise ValueError("Invalid parameter mode = '%s' !!" % self.mode)

def unsorted_segment_sum(data : torch.Tensor, segment_ids : torch.Tensor, num_segments : int) -> torch.Tensor:
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    segment_ids = torch.unsqueeze(segment_ids, 1)
    segment_ids = segment_ids.expand(-1, data.size(1))
    result = result.scatter_add(0, segment_ids, data)
    return result

def unsorted_segment_mean(data : torch.Tensor, segment_ids : torch.Tensor, num_segments : int) -> torch.Tensor:
    r'''Custom PyTorch op to replicate TensorFlow's `unsorted_segment_mean`.
    '''
    result = data.new_zeros((num_segments, data.size(1)))
    count = data.new_zeros((num_segments, data.size(1)))
    segment_ids = torch.unsqueeze(segment_ids, 1)
    segment_ids = segment_ids.expand(-1, data.size(1))
    result = result.scatter_add(0, segment_ids, data)
    count = count.scatter_add(0, segment_ids, torch.ones_like(data))
    result = result / count.clamp(min=1)
    return result
