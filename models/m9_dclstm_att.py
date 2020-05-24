import torch

from models.lightning_common import CommonModel, common_test, common_train
from torch.nn import functional as F


# Architecture based on
# Singh et al.
# https://github.com/isukrit/encodingHumanActivity/blob/master/codes/model_proposed/model_with_self_attn.py
class LSTMModel(CommonModel):
    def __init__(self, hparams, xst, yst, xsv, ysv):
        super().__init__(hparams, xst, yst, xsv, ysv)

        self.hidden_size = 32
        self.cnn_filters = 3
        self.lstm_layers = 1
        self.att_size = 32
        self.att_hops = 10

        self.c1 = torch.nn.Conv2d(1, self.cnn_filters, kernel_size=(1, hparams['channels']))
        self.lstm = torch.nn.LSTM(self.cnn_filters, self.hidden_size, self.lstm_layers, dropout=0, batch_first=True)

        self.self_att_score = torch.nn.Parameter(torch.Tensor(self.att_size, self.hidden_size))
        torch.nn.init.xavier_uniform_(self.self_att_score)
        self.self_att_weight = torch.nn.Parameter(torch.Tensor(self.att_hops, self.att_size))
        torch.nn.init.xavier_uniform_(self.self_att_weight)

        self.fc = torch.nn.Linear(self.hidden_size * self.att_hops, hparams['class_count'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])

        x = self.c1(x)
        x = x.squeeze(3).permute(0, 2, 1)
        x, _ = self.lstm(x)  # -> Batch, length, hidden (=32)

        # Variable naming according to source
        # -> Batch, hidden, length
        hidden_states_transposed = x.permute(0, 2, 1)
        # att_size x hidden bmm hidden x length -> att_size x length
        attention_score = torch.tanh(torch.matmul(self.self_att_score, hidden_states_transposed))
        # att_hops x att_size bmm att_size x length -> att_hops x length
        attention_weights = F.softmax(torch.matmul(self.self_att_weight, attention_score), dim=2)

        # att_hops x length bmm length x hidden = att_hops x hidden
        embedding_matrix = torch.bmm(attention_weights, x)
        # batch x (att_hops * hidden)
        embedding_matrix = embedding_matrix.flatten(start_dim=1)

        x = self.fc(embedding_matrix)
        return x


def train(x_train, y_train, class_count, *args, **kwargs):
    return common_train(x_train, y_train, LSTMModel,
                        {
                            'lr': kwargs['lr'],
                            'class_count': class_count,
                            'channels': x_train.shape[2],
                        },
                        kwargs['folds'])


def test(model, x_test):
    return common_test(model, x_test)
