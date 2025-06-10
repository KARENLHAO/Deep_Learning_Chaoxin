import torch
import torch.nn as nn

class BiGRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=5, bidirectional=True):
        super(BiGRUNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          batch_first=True)

    def forward(self, x):
        output, h_n = self.gru(x)
        return output, h_n

# 示例输入
x = torch.randn(32, 100, 64)  # batch_size=32, seq_len=100, input_size=64
model = BiGRUNet(input_size=64, hidden_size=128)
output, h_n = model(x)
print(output.shape)  # [32, 100, 256] 因为是双向的，128*2=256
