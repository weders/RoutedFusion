import torch


class UNet(torch.nn.Module):
    """
    Basic UNet building block, calling itself recursively.
    Note that the final output does not have a ReLU applied.
    """

    def __init__(self, Cin, F, Cout, depth, batchnorms=True):

        super().__init__()
        self.F = F
        self.depth = depth

        if batchnorms:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(Cout),
            )
        else:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU()
            )

        if depth > 1:
            self.process = UNet(F, 2 * F, 2 * F, depth - 1, batchnorms=batchnorms)
        else:
            if batchnorms:
                self.process = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.BatchNorm2d(2 * F),
                    torch.nn.ReLU(),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(2 * F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.BatchNorm2d(2 * F),
                    torch.nn.ReLU(),
                )
            else:
                self.process = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(2 * F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):

        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]
        upsampled = upsampled[:, :, :H, :W]
        output = self.post(torch.cat((features, upsampled), dim=1))

        return output


class ConfidenceRouting(torch.nn.Module):
    """
    Network for confidence routing in RoutedFusion.
    """

    def __init__(self, Cin, F, Cout, depth, batchnorms=True):

        super().__init__()
        self.F = F
        self.depth = depth

        if batchnorms:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(Cout),
            )
        else:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU()
            )

        if depth > 1:
            self.process = UNet(F, 2 * F, 2 * F, depth - 1, batchnorms=batchnorms)
        else:
            if batchnorms:
                self.process = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.BatchNorm2d(2 * F),
                    torch.nn.ReLU(),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(2 * F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.BatchNorm2d(2 * F),
                    torch.nn.ReLU(),
                )
            else:
                self.process = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                    torch.nn.ReflectionPad2d(1),
                    torch.nn.Conv2d(2 * F, 2 * F, kernel_size=3, stride=1, padding=0),
                    torch.nn.ReLU(),
                )

        self.uncertainty = torch.nn.Sequential(torch.nn.ReflectionPad2d(1),
                                               torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                                               torch.nn.ReLU(),
                                               torch.nn.ReflectionPad2d(1),
                                               torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                                               torch.nn.ReLU())

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):

        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(lower_features, scale_factor=2, mode="bilinear",
                                                    align_corners=False)
        H = data.shape[2]
        W = data.shape[3]
        upsampled = upsampled[:, :, :H, :W]
        output = self.post(torch.cat((features, upsampled), dim=1))
        uncertainty = self.uncertainty(torch.cat((features, upsampled), dim=1))

        return torch.cat((output, uncertainty), dim=1)

    def get_influence_percentages(self):
        """
        This function is intended to return a matrix of influences.
        I.e. for each output channel it returns the percentage it is controlled by each input channel.
        Very very roughly speaking, as all this does is iteratively calculate these percentages based on fractional absolute weighting.
        Output:
            percentages -- C_out x C_in matrix giving the weights
        """
        if isinstance(self.pre[1], torch.nn.BatchNorm2d):
            print("BatchNorm UNets not supported for influence percentages")
            return None
        pre1 = self.pre[1].weight.abs().sum(dim=3).sum(dim=2)
        pre1 = pre1 / pre1.sum(dim=1, keepdim=True)
        pre2 = self.pre[4].weight.abs().sum(dim=3).sum(dim=2)
        pre2 = pre2 / pre2.sum(dim=1, keepdim=True)
        pre2 = torch.matmul(pre2, pre1)
        if isinstance(self.process, UNet):
            process2 = torch.matmul(self.process.get_influence_percentages(), pre2)
        else:
            process1 = self.process[1].weight.abs().sum(dim=3).sum(dim=2)
            process1 = process1 / process1.sum(dim=1, keepdim=True)
            process1 = torch.matmul(process1, pre2)
            process2 = self.process[4].weight.abs().sum(dim=3).sum(dim=2)
            process2 = process2 / process2.sum(dim=1, keepdim=True)
            process2 = torch.matmul(process2, process1)

        post1 = self.post[1].weight.abs().sum(dim=3).sum(dim=2)
        post1 = post1 / post1.sum(dim=1, keepdim=True)
        post1 = torch.matmul(post1, torch.cat((pre2, process2), dim=0))
        post2 = self.post[4].weight.abs().sum(dim=3).sum(dim=2)
        post2 = post2 / post2.sum(dim=1, keepdim=True)
        post2 = torch.matmul(post2, post1)

        return post2
        return final_layer
