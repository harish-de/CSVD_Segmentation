# import torch
# import torch.nn as nn
# import os
# import json
#
# class Conv():
#     # dir_path = os.path.dirname(__file__)
#     # config_path = os.path.relpath('..\\configs\\config.json', dir_path)
#
#     def __init__(self, in_channel, out_channel):
#         super(Conv, self).__init__()
#
#         # with open(self.config_path) as json_file:
#         #     data = json.load(json_file)
#         #     model = data['model']
#         #     conv = model['conv']
#         #     kernel_size = conv['kernel_size']
#         #     padding = conv['padding']
#         #     stride = conv['stride']
#         kernel_size = 3
#         padding = 1
#         stride = 1
#
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                       kernel_size=kernel_size, padding=padding, stride=stride),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
#                       kernel_size=kernel_size, padding=padding, stride=stride),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self,x):
#         return self.conv_block(x);
#
# class Max():
#     def __init__(self):
#         super(Max, self).__init__()
#         # with open(self.config_path) as json_file:
#         #     data = json.load(json_file)
#         #     model = data['model']
#         #     conv = model['max_pooling_upConvTranspose']
#         #     kernel_size = conv['kernel_size']
#         #     stride = conv['stride']
#         kernel_size = 2
#         stride = 2
#
#         self.max_pool = nn.MaxPool2d(kernel_size= kernel_size, stride=stride)
#
#     def forward(self,x):
#         return self.max_pool(x)
#
# class upTrans():
#     def __init__(self, in_channel, out_channel):
#         super(upTrans, self).__init__()
#         # with open(self.config_path) as json_file:
#         #     data = json.load(json_file)
#         #     model = data['model']
#         #     conv = model['max_pooling_upConvTranspose']
#         #     kernel_size = conv['kernel_size']
#         #     stride = conv['stride']
#         kernel_size = 2
#         stride = 2
#
#         self.up_trans = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride)
#
#     def forward(self,x):
#         return self.up_trans(x)
#
# class ZeroPad():
#     def __init__(self):
#         super(ZeroPad, self).__init__()
#         self.zero_pad = nn.ZeroPad2d(3)
#
#     def forward(self,x):
#         return self.zero_pad(x)
#
# class convFinal():
#     def __init__(self,in_channel, out_channel):
#         super(convFinal, self).__init__()
#
#         self.conv_final = nn.Sequential(
#             nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
#                       kernel_size=1, padding=(1, 1), stride=1),
#             torch.nn.Sigmoid())
#
#     def forward(self,x):
#         return self.conv_final(x)
#
#
# class unet(nn.Module):
#     def __init__(self):
#         super(unet, self).__init__()
#
#         self.down1 = Conv(2,64).conv_block
#         self.max1 = Max().max_pool
#         self.down2 = Conv(64,96).conv_block
#         self.max2 = Max().max_pool
#         self.down3 = Conv(96,128).conv_block
#         self.max3 = Max().max_pool
#         self.down4 = Conv(128,256).conv_block
#         self.max4 = Max().max_pool
#         self.down5 = Conv(256,512).conv_block
#         self.up_trans1 = upTrans(512,512).up_trans
#         self.up1 = Conv(768,256).conv_block
#         self.up_trans2 = upTrans(256,256).up_trans
#         self.up2 = Conv(384,128).conv_block
#         self.up_trans3 = upTrans(128,128).up_trans
#         self.up3 = Conv(224,96).conv_block
#         self.up_trans4 = upTrans(96,96).up_trans
#         self.up4 = Conv(160,64).conv_block
#         self.zeroPad = ZeroPad().zero_pad
#         self.final_conv = convFinal(64,1).conv_final
#
#     def forward(self,x):
#         x = self.down1(x)
#         conv1_out = x
#         conv1_dim = x.shape[2]
#         x = self.max1(x)
#
#         x = self.down2(x)
#         conv2_out = x
#         conv2_dim = x.shape[2]
#         x = self.max2(x)
#
#         x = self.down3(x)
#         conv3_out = x
#         conv3_dim = x.shape[2]
#         x = self.max3(x)
#
#         x = self.down4(x)
#         conv4_out = x
#         conv4_dim = x.shape[2]
#         x = self.max4(x)
#
#         x = self.down5(x)
#
#         x = self.up_trans1(x)
#         lower = int((conv4_dim - x.shape[2]) / 2)
#         upper = int(conv4_dim - lower)
#         conv4_out_modified = conv4_out[:, :, 0:x.shape[2], 0:x.shape[3]]
#         x = torch.cat([x, conv4_out_modified], dim=1)
#         x = self.up1(x)
#
#         x = self.up_trans2(x)
#         lower = int((conv3_dim - x.shape[2]) / 2)
#         upper = int(conv3_dim - lower)
#         conv3_out_modified = conv3_out[:, :, 0:x.shape[2], 0:x.shape[3]]
#         x = torch.cat([x, conv3_out_modified], dim=1)
#         x = self.up2(x)
#
#         x = self.up_trans3(x)
#         lower = int((conv2_dim - x.shape[2]) / 2)
#         upper = int(conv2_dim - lower)
#         conv2_out_modified = conv2_out[:, :, 0:x.shape[2], 0:x.shape[3]]
#         x = torch.cat([x, conv2_out_modified], dim=1)
#         x = self.up3(x)
#
#         x = self.up_trans4(x)
#         lower = int((conv1_dim - x.shape[2]) / 2)
#         upper = int(conv1_dim - lower)
#         conv1_out_modified = conv1_out[:, :, 0:x.shape[2], 0:x.shape[3]]
#         x = torch.cat([x, conv1_out_modified], dim=1)
#         x = self.up4(x)
#
#         x = self.zeroPad(x)
#
#         x = self.final_conv(x)
#
#         return x

import torch
import torch.nn as nn

class CleanU_Net(nn.Module):

    def __init__(self):

        super(CleanU_Net, self).__init__()

        # Conv block 1 - Down 1
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=64,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),

        )
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 2 - Down 2
        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 3 - Down 3
        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 4 - Down 4
        self.conv4_block = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv block 5 - Down 5
        self.conv5_block = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 1
        self.up_1 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)

        # Up Conv block 1
        self.conv_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=256,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 2
        self.up_2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)

        # Up Conv block 2
        self.conv_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 3
        self.up_3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)

        # Up Conv block 3
        self.conv_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=224, out_channels=96,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
        )

        # Up 4
        self.up_4 = nn.ConvTranspose2d(in_channels=96, out_channels=96, kernel_size=2, stride=2)

        # Up Conv block 4
        self.conv_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=64,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=3, padding=(1,1), stride=1),
            nn.ReLU(inplace=True),
            nn.ZeroPad2d(3)
        )

        self.conv_final = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=1,
                                    kernel_size=1, padding=(1,1), stride=1),
                            torch.nn.Sigmoid())



    def forward(self, x):
        # print('input', x.shape)

        # Down 1
        x = self.conv1_block(x)
        # print('after conv1', x.shape)
        conv1_out = x  # Save out1
        conv1_dim = x.shape[2]
        x = self.max1(x)
        # print('before conv2', x.shape)

        # Down 2
        x = self.conv2_block(x)
        # print('after conv2', x.shape)
        conv2_out = x
        conv2_dim = x.shape[2]
        x = self.max2(x)
        # print('before conv3', x.shape)

        # Down 3
        x = self.conv3_block(x)
        # print('after conv3', x.shape)
        conv3_out = x
        conv3_dim = x.shape[2]
        x = self.max3(x)
        # print('before conv4', x.shape)

        # Down 4
        x = self.conv4_block(x)
        # print('after conv5', x.shape)
        conv4_out = x
        conv4_dim = x.shape[2]
        x = self.max4(x)

        # Midpoint
        x = self.conv5_block(x)

        # Up 1
        x = self.up_1(x)
        # print('up_1', x.shape)
        lower = int((conv4_dim - x.shape[2])/2)
        upper = int(conv4_dim - lower)

        conv4_out_modified = conv4_out[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.cat([x, conv4_out_modified], dim=1)


        # print('after cat_1', x.shape)
        x = self.conv_up_1(x)
        # print('after conv_1', x.shape)

        # Up 2
        x = self.up_2(x)
        # print('up_2', x.shape)
        lower = int((conv3_dim - x.shape[2]) / 2)
        upper = int(conv3_dim - lower)
        conv3_out_modified = conv3_out[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.cat([x, conv3_out_modified], dim=1)
        # print('after cat_2', x.shape)
        x = self.conv_up_2(x)
        # print('after conv_2', x.shape)

        # Up 3
        x = self.up_3(x)
        # print('up_3', x.shape)
        lower = int((conv2_dim - x.shape[2]) / 2)
        upper = int(conv2_dim - lower)
        conv2_out_modified = conv2_out[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.cat([x, conv2_out_modified], dim=1)
        # print('after cat_3', x.shape)
        x = self.conv_up_3(x)
        # print('after conv_3', x.shape)

        # Up 4
        x = self.up_4(x)
        # print('up_4', x.shape)
        lower = int((conv1_dim - x.shape[2]) / 2)
        upper = int(conv1_dim - lower)
        conv1_out_modified = conv1_out[:, :, 0:x.shape[2], 0:x.shape[3]]
        x = torch.cat([x, conv1_out_modified], dim=1)
        # print('after cat_4', x.shape)
        x = self.conv_up_4(x)
        # print('after conv_4', x.shape)

        # Final output
        x = self.conv_final(x)

        return x


