# pnnx model stat
# model inputshape = [1,52]f32
# FLOPS = 16.389K
# memory OPS = 8.446K

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.F_linear_0 = nn.Linear(bias=True, in_features=52, out_features=64)
        self.F_linear_1 = nn.Linear(bias=True, in_features=64, out_features=64)
        self.F_linear_2 = nn.Linear(bias=True, in_features=64, out_features=5)

        archive = zipfile.ZipFile('gridworld.pnnx.bin', 'r')
        self.F_linear_0.bias = self.load_pnnx_bin_as_parameter(archive, 'F_linear_0.bias', (64), 'float32')
        self.F_linear_0.weight = self.load_pnnx_bin_as_parameter(archive, 'F_linear_0.weight', (64,52), 'float32')
        self.F_linear_1.bias = self.load_pnnx_bin_as_parameter(archive, 'F_linear_1.bias', (64), 'float32')
        self.F_linear_1.weight = self.load_pnnx_bin_as_parameter(archive, 'F_linear_1.weight', (64,64), 'float32')
        self.F_linear_2.bias = self.load_pnnx_bin_as_parameter(archive, 'F_linear_2.bias', (5), 'float32')
        self.F_linear_2.weight = self.load_pnnx_bin_as_parameter(archive, 'F_linear_2.weight', (5,64), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.F_linear_0(v_0)
        v_2 = F.tanh(v_1)
        v_3 = self.F_linear_1(v_2)
        v_4 = F.tanh(v_3)
        v_5 = self.F_linear_2(v_4)
        return v_5

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 52, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("gridworld_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 52, dtype=torch.float)

    torch.onnx.export(net, v_0, "gridworld_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 52, dtype=torch.float)

    import pnnx
    pnnx.export(net, "gridworld_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 52, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
