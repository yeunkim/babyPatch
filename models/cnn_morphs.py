
import torch
import torch.nn as nn


# create DQN framework
# (1) create one model for dilation
class dilation_nn(nn.Module):
    def __init__(self, h, w, actions, channels=1, channelsL=3):
        super(dilation_nn, self).__init__()
        self.channels = channels
        self.channelsL = channelsL
        self.h = h
        self.w = w
        # self.iters = iters
        # self.op_shape = op_shape
        self.actions = actions
        self.f1 = 32
        self.f2 = 64
        self.f3 = 128
        self.kw1 = 2
        self.kw2 = 3
        self.kw3 = 5
        self.stride = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f3, self.kw3),
            nn.BatchNorm2d(self.f3),
            nn.LeakyReLU(0.1, True)
        )
        self.convL = nn.Sequential(
            nn.Conv2d(self.channelsL, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f3, self.kw3),
            nn.BatchNorm2d(self.f3),
            nn.LeakyReLU(0.1, True)
        )

        def conv2d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, self.kw1), self.kw2), self.kw3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, self.kw1), self.kw2), self.kw3)
        linear_input_size = convw * convh * self.f2

        h1 = 256
        h2 = 100
        # self.head = nn.Sequential(nn.Linear(linear_input_size, h2),
        #                           nn.BatchNorm1d(h2),
        #                           nn.LeakyReLU(0.1, True),
        #                           nn.Linear(h2, self.iters)
        #                           )
        linear_input_size = 2048 *2
        self.head2 = nn.Sequential(nn.Linear(linear_input_size, h1),
                                  # nn.BatchNorm1d(h1),
                                  nn.LeakyReLU(0.1, True),
                                   nn.Linear(h1, h1),
                                   nn.LeakyReLU(0.1, True),
                                  nn.Linear(h1, h2),
                                   nn.LeakyReLU(0.1, True),
                                   nn.Linear(h2, self.actions),
                                   nn.Softmax(dim=1)
                                  )

    def forward(self, img, label):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        L = self.convL(label)
        L = L.view(L.size(0), -1)

        out = torch.cat((x,L), dim=1)
        iters = self.head2(out.view(out.size(0), -1))

        # turn off shape choices for now
        # shape = self.head2(out.view(out.size(0), -1))

        return iters # , shape
# (2) create one model for erosion
class erosion_nn(nn.Module):
    def __init__(self, h, w, erode, channels=1):
        super(erosion_nn, self).__init__()
        self.channels = channels
        self.h = h
        self.w = w
        self.erode = erode
        self.f1 = 32
        self.f2 = 64
        self.kw1 = 2
        self.kw2 = 3
        self.kw3 = 5
        self.stride = 1

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f2, self.kw3),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True)
        )
        self.convL = nn.Sequential(
            nn.Conv2d(self.channels, self.f1, self.kw1, self.stride),
            nn.BatchNorm2d(self.f1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f1, self.f2, self.kw2),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(self.f2, self.f2, self.kw3),
            nn.BatchNorm2d(self.f2),
            nn.LeakyReLU(0.1, True)
        )

        def conv2d_size_out(size, kernel_size=5, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, self.kw1), self.kw2), self.kw3)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, self.kw1), self.kw2), self.kw3)
        linear_input_size = convw * convh * self.f2

        h1 = 256
        h2 = 100
        self.head = nn.Sequential(nn.Linear(linear_input_size, h1),
                                  nn.BatchNorm1d(h1),
                                  nn.LeakyReLU(0.1, True),
                                  nn.Linear(h1, self.erode)
                                  )


    def forward(self, img, label):
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        L = self.convL(label)
        L = L.view(L.size(0), -1)

        out = x*L
        erodes = self.head(out.view(out.size(0), -1))

        return erodes