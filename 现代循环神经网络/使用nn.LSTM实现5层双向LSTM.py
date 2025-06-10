import torch
import torch.nn as nn

class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=5, bidirectional=True):
        super(BiLSTMNet, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            batch_first=True)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return output, h_n, c_n

# 示例输入
x = torch.randn(32, 100, 64)  # batch_size=32, seq_len=100, input_size=64
model = BiLSTMNet(input_size=64, hidden_size=128)
output, h_n, c_n = model(x)
print(output.shape)  # [32, 100, 256]
