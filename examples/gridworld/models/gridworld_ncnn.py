import numpy as np
import ncnn
import torch

def test_inference():
    torch.manual_seed(0)
    in0 = torch.rand(1, 52, dtype=torch.float)
    out = []

    with ncnn.Net() as net:
        net.load_param("gridworld.ncnn.param")
        net.load_model("gridworld.ncnn.bin")

        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(in0.numpy()).clone())

            _, out0 = ex.extract("out0")
            out.append(torch.from_numpy(np.array(out0)))

    if len(out) == 1:
        return out[0]
    else:
        return tuple(out)

if __name__ == "__main__":
    print(test_inference())
