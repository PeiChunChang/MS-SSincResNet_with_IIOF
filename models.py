class SquaredSincConv_fast(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def plot_triFilter(center, band):
        num_band = center.shape[0]
        low = center - band
        high = center + band
        for i in range(num_band):
            p1 = low[i]
            c = center[i]
            p2 = high[i]
            plt.plot([p1, c, p2], [0, 1, 0])
        plt.show()

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):
        super().__init__()

        if in_channels != 1:
            #msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate

        # initialize filterbanks such that they are equally spaced in Mel scale
        self.min_hz = 0
        self.max_hz = self.sample_rate / 2
        mel_min = self.to_mel(self.min_hz)
        mel_max = self.to_mel(self.max_hz)
        delta_mel = np.abs(mel_max - mel_min) / ( self.out_channels + 1.0)
        frequencies_mel = mel_min + delta_mel*np.arange(0, self.out_channels+2)
        lower_edges = self.to_hz(frequencies_mel[:-2])
        upper_edges = self.to_hz(frequencies_mel[2:])
        center_frequencies = self.to_hz(frequencies_mel[1:-1])
        band_frequencies = (upper_edges - lower_edges)/2

        # filter center frequency (out_channels, 1)
        self.center_hz_ = nn.Parameter(torch.Tensor(center_frequencies).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(band_frequencies).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2))) # computing only half of the window
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size);

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = torch.clamp(self.center_hz_ - self.band_hz_, self.min_hz, self.max_hz)
        high = torch.clamp(self.center_hz_ + self.band_hz_, self.min_hz, self.max_hz)
        band = high - low

        f_times_t_center = torch.matmul((high + low) / 2, self.n_)
        f_times_t_band = torch.matmul(band, self.n_) * 0.5 * 0.5

        band_pass_left = ((torch.sin(f_times_t_band)/(f_times_t_band)) ** 2 * torch.cos(f_times_t_center)) * self.window_
        band_pass_center = torch.ones(self.out_channels, 1).to(waveforms.device)
        band_pass_right= torch.flip(band_pass_left, dims=[1])

        band_pass=torch.cat([band_pass_left, band_pass_center,band_pass_right], dim=1)
        self.filters = (band_pass).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)
class myResnet(nn.Module):
    def __init__(self, pretrained=True):
        super(myResnet5, self).__init__()
        self.resnet18_layer3 = nn.Sequential(
                    *list(models.resnet18(pretrained=True).children())[:-3]
                )
        self.resnet18_global = nn.Sequential(
                    *list(models.resnet18(pretrained=True).children())[7:9]
                )
    def forward(self, x):
        feature_layer3 = self.resnet18_layer3(x)
        feature_global = self.resnet18_global(feature_layer3)
        return feature_layer3, feature_global
class myResnet(nn.Module):
    def __init__(self, pretrained=True):
        super(myResnet5, self).__init__()
        self.resnet18_layer3 = nn.Sequential(
                    *list(models.resnet18(pretrained=True).children())[:-3]
                )
        self.resnet18_global = nn.Sequential(
                    *list(models.resnet18(pretrained=True).children())[7:9]
                )
    def forward(self, x):
        feature_layer3 = self.resnet18_layer3(x)
        feature_global = self.resnet18_global(feature_layer3)
        return feature_layer3, feature_global
class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat, concat=True):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)

        projection = torch.bmm(global_feat.unsqueeze(1), torch.flatten(local_feat, start_dim=2))
        projection = torch.bmm(global_feat.unsqueeze(2), projection).view(local_feat.size())

        projection = projection / (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        if concat:
            return torch.cat([global_feat.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)
        else:
            return orthogonal_comp
class MS_SSincResNet_IIOF(nn.Module):
    def __init__(self):
        super().__init__()
        self.layerNorm = nn.LayerNorm([1, 144000])
        self.sincNet1 = nn.Sequential(
            SquaredSincConv_fast(out_channels=160, kernel_size=251),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(2048))
        self.sincNet2 = nn.Sequential(
            SquaredSincConv_fast(out_channels=160, kernel_size=501),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(2048))
        self.sincNet3 = nn.Sequential(
            SquaredSincConv_fast(out_channels=160, kernel_size=1001),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(2048))

        self.resnet = myResnet(pretrained=True)
        self.spp = SPP()
        self.of = OrthogonalFusion()
        self.fc = nn.Linear(512*2*9, 2, bias=True)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.layerNorm(x)
        x = torch.cat((self.sincNet1(x).unsqueeze_(dim=1),
                       self.sincNet2(x).unsqueeze_(dim=1),
                       self.sincNet3(x).unsqueeze_(dim=1)), dim=1)
        local_feat, global_feat = self.resnet(x)

        local_feat = self.spp(local_feat)
        (h1, h2) = torch.split(global_feat, 256, dim=1)
        h2 = h2.squeeze()

        h1_ = self.of(h1, h2, concat=False).squeeze()
        x1 = self.of(local_feat, h1_)
        x2 = self.of(local_feat, h2)

        x = torch.cat([x1, x2], dim=1).view(x.size()[0], -1)
        x = self.fc(x)
        x = self.tanh(x)

        return x