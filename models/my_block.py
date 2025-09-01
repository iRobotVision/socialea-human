import torch
from torch import nn
import torch.nn.functional as F

    
class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1):
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.conv_layer = nn.Conv1d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                    kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, c, l = x.size()

        ffted = torch.fft.rfft(x, norm='ortho')

        x_fft_real = torch.real(ffted)
        x_fft_imag = torch.imag(ffted)

        ffted = torch.stack((x_fft_real, x_fft_imag), dim=-1)
        ffted = ffted.permute(0, 1, 3, 2).contiguous()
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 2).contiguous()
        ffted = torch.view_as_complex(ffted)

        output = torch.fft.irfft(ffted, n=l, norm='ortho')

        return output


class Freq_Fusion(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1, 3, 5, 7],
            se_ratio=4,
            local_size=8,
            scale_ratio=2,
            spilt_num=4
    ):
        super(Freq_Fusion, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.size = local_size
        self.dim_sp = dim*scale_ratio//spilt_num

        self.conv_init_1 = nn.Sequential(  # PW
            nn.Conv1d(dim, dim, 1),
            nn.GELU()
        )
        self.conv_init_2 = nn.Sequential(  # DW
            nn.Conv1d(dim, dim, 1),
            nn.GELU()
        )

        # Fourier
        self.FFC = FourierUnit(self.dim*2, self.dim*2)
        self.bn = nn.BatchNorm1d(dim*2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x_1, x_2 = torch.split(x, self.dim, dim=1)
        x_1 = self.conv_init_1(x_1)
        x_2 = self.conv_init_2(x_2)

        x0 = torch.cat([x_1, x_2], dim=1)
     
        x = self.FFC(x0) + x0
        x = self.relu(self.bn(x))

        return x


class FCME(nn.Module):
    def __init__(
            self,
            dim,
            token_mixer_for_gloal=Freq_Fusion,
            mixer_kernel_size=[1, 3, 5, 7],
            local_size=8
    ):
        super(FCME, self).__init__()
        self.dim = dim

        self.mixer_gloal = token_mixer_for_gloal(dim=self.dim, kernel_size=mixer_kernel_size,
                                                 se_ratio=8, local_size=local_size)
        self.ca_conv = nn.Sequential(
            nn.Conv1d(2*dim, dim, 1),
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.dw_conv_1 = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, kernel_size=3, padding=1,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.dw_conv_2 = nn.Sequential(
            nn.Conv1d(self.dim, self.dim, kernel_size=7, padding=3,
                      groups=self.dim, padding_mode='reflect'),
            nn.GELU()
        )

    def forward(self, x):

        x_local_1 = self.dw_conv_1(x)
        x_local_2 = self.dw_conv_2(x)
        x_gloal = self.mixer_gloal(torch.cat([x_local_1, x_local_2], dim=1))

        x = self.ca_conv(x_gloal)
        x = self.ca(x) * x
        return x


class TCCA(nn.Module):
    def __init__(self, K, d, kernel_size):
        super(TCCA, self).__init__()
        D = K * d
        self.d = d
        self.K = K
        self.FC_v = nn.Linear(D,D)
        self.FC = nn.Linear(D,D)
        self.kernel_size = kernel_size
        self.padding = self.kernel_size-1
        self.cnn_q = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.cnn_k = nn.Conv2d(D, D, (1, self.kernel_size), padding=(0, self.padding))
        self.norm_q = nn.BatchNorm2d(D)
        self.norm_k = nn.BatchNorm2d(D)
    def forward(self, X):
        # (B,T,N,D)
        batch_size = X.shape[0]

        X_ = X.permute(0, 3, 2, 1) # (B,T,N,D) --> (B,D,N,T)

        # key: (B,T,N,D)  value: (B,T,N,D)
        query = self.norm_q(self.cnn_q(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1) # (B,D,N,T)--permute-->(B,T,N,D)
        key = self.norm_k(self.cnn_k(X_))[:, :, :, :-self.padding].permute(0, 3, 2, 1) #  (B,D,N,T)--permute-->(B,T,N,D)
        value = self.FC_v(X) # (B,T,N,D)-->(B,T,N,D)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0) 
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)  
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0) 

        query = query.permute(0, 2, 1, 3)  # query: (B*k,T,N,d) --> (B*k,N,T,d)
        key = key.permute(0, 2, 3, 1)      # key: (B*k,T,N,d) --> (B*k,N,d,T)
        value = value.permute(0, 2, 1, 3)  # key: (B*k,T,N,d) --> (B*k,N,T,d)

        attention = (query @ key) * (self.d ** -0.5) # (B*k,N,T,d) @ (B*k,N,d,T) = (B*k,N,T,T)
        attention = F.softmax(attention, dim=-1) 

        X = (attention @ value) #  (B*k,N,T,T) @ (B*k,N,T,d) = (B*k,N,T,d)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1) # (B*k,N,T,d)-->(B,N,T,d*k)==(B,N,T,D)
        X = self.FC(X) 

        
        return X.permute(0, 2, 1, 3) # (B,N,T,D)-->(B,T,N,D)
    