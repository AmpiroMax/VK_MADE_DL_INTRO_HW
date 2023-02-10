""" Model achitecture """

from torch import nn


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 1
        out_channels = 512
        activation_map_height = 1
        cnn_embedding_size = 64
        hidden_rnn_size = 256
        num_classes = 37

        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(
                in_channels=512,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(
                kernel_size=(2, 1)
            ),
            nn.Conv2d(
                in_channels=512,
                out_channels=out_channels,
                kernel_size=2,
                stride=1,
                padding=0
            ),
        )

        self.map_2_seq_layer = nn.Sequential(
            nn.Linear(
                out_channels * activation_map_height,
                cnn_embedding_size
            )
        )

        self.bi_rnn_layer = nn.Sequential(
            nn.LSTM(
                input_size=cnn_embedding_size,
                hidden_size=hidden_rnn_size,
                num_layers=2,
                bidirectional=True
            )
        )

        self.transcription_layer = nn.Sequential(
            nn.Linear(
                in_features=hidden_rnn_size * 2,
                out_features=num_classes
            )
        )

        self.conv_layer.apply(self._init_weights)
        self.map_2_seq_layer.apply(self._init_weights)
        self.bi_rnn_layer.apply(self._init_weights)
        self.transcription_layer.apply(self._init_weights)

    def forward(self, images):
        activation_maps = self.conv_layer(images)
        batch, channel, height, width = activation_maps.size()

        activation_maps = activation_maps.view(batch, channel * height, width)

        activation_maps = activation_maps.permute(2, 0, 1)
        cnn_embedding_seq = self.map_2_seq_layer(activation_maps)

        rnn_embeddings_seq, _ = self.bi_rnn_layer(cnn_embedding_seq)

        logits_seq = self.transcription_layer(rnn_embeddings_seq)
        return logits_seq

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

        if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


if __name__ == "__main__":
    model = RCNN()
    print(model)
