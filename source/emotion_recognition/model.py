# NN architecture

from torch import nn


class emotion_cnn(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        # extract features of the image using, convolution, activation, pooling
        self.features = nn.Sequential(
            #initial extraction
            nn.Conv2d(                          # CONVOLUTION:
                in_channels = 1,                # 1 channel, because inherit greyscale + sure greyscale after augmentation
                out_channels = 32,              # no. of filters
                kernel_size = 3,                # filter dimensions - n x n
                padding = 1,                    # border length, or literally padding
            ),
                                                #TODO: make below comment better
            nn.ReLU(),                          #ACTIVATION: negative nimber -> 0, introduce nonlinear(?)
            nn.MaxPool2d(kernel_size = 2),      # POOLING:take max value out of n x n in the consideration, reduce cost bit also reduce details
                                                # 48 x 48 -> 24 x 24
            # additional features from initial extraction
            nn.Conv2d(
                in_channels = 32,     # 1 -> 32
                out_channels = 64,    # 32 -> 64
                kernel_size = 3,
                padding = 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),      # 24 x 24 -> 12 x 12

            #Further features form previous extraction
            nn.Conv2d(
                in_channels = 64,     # 32 -> 64
                out_channels = 128,   # 64 -> 128
                kernel_size = 3,
                padding = 1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),      # 12 x 12 -> 6 x 6
        )   
        #convert feature map to vector
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features = 128 * 6 * 6,
                out_features = 256,
            ),
            nn.ReLU(),

        nn.Dropout(p=0.4),                      # set to 0 activations at 0.n * 100% rate to reduce overfitting

        # create scores for each emmotion classification/class
        nn.Linear(
            in_features = 256,
            out_features = num_classes,
        ),
    )
    # pass inputs thru feature extraction and return class scores
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

