import torch
import torch.nn as nn

from Recursive_KD import HL_RecursiveKD, Mid_Ht_Intra_Fusion
from DiffKD import DiffKD_Ht


def cosine_similarity_loss(teacher_features, student_features, eps=1e-6):
    tea_avg = torch.mean(teacher_features, dim=2, keepdim=True)
    tea_FW_mask = torch.sigmoid(tea_avg).permute(1, 0, 2)

    student_features = student_features.permute(1, 0, 2)
    student_norm = torch.sqrt(
        torch.sum(student_features ** 2, dim=2, keepdim=True)
    )
    student_features = student_features / (student_norm + eps)
    student_s = torch.bmm(student_features, student_features.permute(0, 2, 1))
    student_s = (student_s + 1.0) / 2.0

    teacher_features = teacher_features.permute(1, 0, 2)
    teacher_norm = torch.sqrt(
        torch.sum(teacher_features ** 2, dim=2, keepdim=True)
    )
    teacher_features = teacher_features / (teacher_norm + eps)
    teacher_s = torch.bmm(teacher_features, teacher_features.permute(0, 2, 1))
    teacher_s = (teacher_s + 1.0) / 2.0

    teacher_s = teacher_s / torch.sum(teacher_s, dim=2, keepdim=True)
    student_s = student_s / torch.sum(student_s, dim=2, keepdim=True)

    loss = (
        (teacher_s - student_s)
        * (torch.log(teacher_s + eps) - torch.log(student_s + eps))
        * tea_FW_mask
    )
    loss = torch.mean(loss, dim=[1, 2]).sum(0)
    return loss


def reshape_for_frame_kd_loss(feature):
    batch_size, channels, timesteps, freq_bins = feature.shape
    feature = feature.permute(0, 2, 1, 3)
    return torch.reshape(feature, [batch_size, timesteps, channels * freq_bins])


class TrainerKDStepDemo(nn.Module):
    """Demo for the KD_loss part in trainer/trainer.py."""

    def __init__(self):
        super(TrainerKDStepDemo, self).__init__()

        self.KD_Block_Mid = nn.Sequential(
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=64,
                trans_stu=True,
            )
        )

        self.KD_Block_Enc = nn.Sequential(
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=129,
            ),
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=64,
            ),
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=64,
            ),
        )

        self.KD_Block_Dec = nn.Sequential(
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=129,
            ),
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=64,
            ),
        )

        self.KD_Block_Ht = nn.Sequential(
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=129,
            ),
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=64,
            ),
            DiffKD_Ht(
                student_channels=64,
                teacher_channels=128,
                kernel_size=3,
                use_ae=True,
                ae_channels=128,
                F_shape=64,
            ),
        )

        mid_channel_stu = 128
        mid_channel_tea = 128

        self.Fusion_Block_Mid_stu = Mid_Ht_Intra_Fusion(
            in_channels=[64, 64],
            out_channels=[64, 64],
            mid_channel=mid_channel_stu,
            shapes=[64, 64],
            detach=False,
        )
        self.Fusion_Block_Mid_tea = Mid_Ht_Intra_Fusion(
            in_channels=[128, 128, 128, 128],
            out_channels=[128, 128, 128, 128],
            mid_channel=mid_channel_tea,
            shapes=[64, 64, 64, 64],
            detach=True,
        )

        mid_channel = 128
        self.Fusion_Block_Enc_stu = HL_RecursiveKD(
            in_channels=[64, 64, 64],
            out_channels=[128, 128, 128],
            mid_channel=mid_channel,
            shapes=[64, 64, 129],
        )
        self.Fusion_Block_Ht_Enc_stu = HL_RecursiveKD(
            in_channels=[64, 64, 64],
            out_channels=[128, 128, 128],
            mid_channel=mid_channel,
            shapes=[64, 64, 129],
        )
        self.Fusion_Block_Dec_stu = HL_RecursiveKD(
            in_channels=[64, 64],
            out_channels=[128, 128],
            mid_channel=mid_channel,
            shapes=[64, 129],
        )

    @staticmethod
    def diffkd_res_loss(kd_block, fs, ft, noise_complex, ht_repeat):
        fs_fea, ft_fea, diff_loss, ae_loss = kd_block(
            fs, ft, noise_complex, ht_repeat
        )

        fs_fea = reshape_for_frame_kd_loss(fs_fea)
        ft_fea = reshape_for_frame_kd_loss(ft_fea)

        if ae_loss != None:
            pred_res_loss = (
                cosine_similarity_loss(ft_fea, fs_fea)
                + diff_loss
                + ae_loss
            )
        else:
            pred_res_loss = (
                cosine_similarity_loss(ft_fea, fs_fea)
                + diff_loss
            )
        return pred_res_loss

    def forward(self, demo_inputs):
        noise_complex = demo_inputs["noise_complex"]
        ht_repeat = demo_inputs["ht_repeat"]

        mid_stu_fea = demo_inputs["mid_stu_fea"]
        mid_tea_fea_list = demo_inputs["mid_tea_fea_list"]
        enc_stu_fea_list = demo_inputs["enc_stu_fea_list"]
        enc_tea_fea_list = demo_inputs["enc_tea_fea_list"]
        dec_stu_fea_list = demo_inputs["dec_stu_fea_list"]
        dec_tea_fea_list = demo_inputs["dec_tea_fea_list"]
        enc_ht_stu_list = demo_inputs["enc_ht_stu_list"]
        enc_ht_tea_list = demo_inputs["enc_ht_tea_list"]

        tmp_mid_stu_fea = mid_stu_fea[::-1]
        tmp_mid_tea_fea = mid_tea_fea_list[::-1]
        mid_fusion_out_stu = self.Fusion_Block_Mid_stu(
            tmp_mid_stu_fea, ht_repeat
        )
        mid_fusion_out_tea = self.Fusion_Block_Mid_tea(
            tmp_mid_tea_fea, ht_repeat
        )

        pred_res_mid_loss = self.diffkd_res_loss(
            self.KD_Block_Mid[0],
            mid_fusion_out_stu,
            mid_fusion_out_tea,
            noise_complex,
            ht_repeat,
        )

        enc_stu_res_list = self.Fusion_Block_Enc_stu(
            enc_stu_fea_list, ht_repeat
        )
        pred_res_enc_loss = 0.0
        for index, (fs, ft) in enumerate(
            zip(enc_stu_res_list, enc_tea_fea_list)
        ):
            pred_res_enc_loss += self.diffkd_res_loss(
                self.KD_Block_Enc[index],
                fs,
                ft,
                noise_complex,
                ht_repeat,
            )

        enc_ht_stu_res_list = self.Fusion_Block_Ht_Enc_stu(
            enc_ht_stu_list, ht_repeat
        )
        pred_res_enc_ht_loss = 0.0
        for index, (fs, ft) in enumerate(
            zip(enc_ht_stu_res_list, enc_ht_tea_list)
        ):
            pred_res_enc_ht_loss += self.diffkd_res_loss(
                self.KD_Block_Ht[index],
                fs,
                ft,
                noise_complex,
                ht_repeat,
            )

        dec_stu_fea_list = dec_stu_fea_list[:-1]
        dec_tea_fea_list = dec_tea_fea_list[:-1]
        dec_stu_fea_list = dec_stu_fea_list[::-1]
        dec_stu_res_list = self.Fusion_Block_Dec_stu(
            dec_stu_fea_list, ht_repeat
        )
        dec_tea_fea_list = dec_tea_fea_list[::-1]

        pred_res_dec_loss = 0.0
        for index, (fs, ft) in enumerate(
            zip(dec_stu_res_list, dec_tea_fea_list)
        ):
            pred_res_dec_loss += self.diffkd_res_loss(
                self.KD_Block_Dec[index],
                fs,
                ft,
                noise_complex,
                ht_repeat,
            )

        KD_loss = (
            pred_res_mid_loss
            + pred_res_enc_loss
            + pred_res_dec_loss
            + pred_res_enc_ht_loss
        )

        return {
            "pred_res_mid_loss": pred_res_mid_loss,
            "pred_res_enc_loss": pred_res_enc_loss,
            "pred_res_dec_loss": pred_res_dec_loss,
            "pred_res_enc_ht_loss": pred_res_enc_ht_loss,
            "KD_loss": KD_loss,
        }


def make_random_inputs(device):
    batch_size = 2
    timesteps = 32

    return {
        "noise_complex": torch.randn(batch_size, 2, timesteps, 257, device=device),
        "ht_repeat": torch.randn(batch_size, 1, timesteps, 257, device=device),
        "mid_stu_fea": [
            torch.randn(batch_size, 64, timesteps, 64, device=device),
            torch.randn(batch_size, 64, timesteps, 64, device=device),
        ],
        "mid_tea_fea_list": [
            torch.randn(batch_size, 128, timesteps, 64, device=device)
            for _ in range(4)
        ],
        "enc_stu_fea_list": [
            torch.randn(batch_size, 64, timesteps, 129, device=device),
            torch.randn(batch_size, 64, timesteps, 64, device=device),
            torch.randn(batch_size, 64, timesteps, 64, device=device),
        ],
        "enc_tea_fea_list": [
            torch.randn(batch_size, 128, timesteps, 129, device=device),
            torch.randn(batch_size, 128, timesteps, 64, device=device),
            torch.randn(batch_size, 128, timesteps, 64, device=device),
        ],
        "enc_ht_stu_list": [
            torch.randn(batch_size, 64, timesteps, 129, device=device),
            torch.randn(batch_size, 64, timesteps, 64, device=device),
            torch.randn(batch_size, 64, timesteps, 64, device=device),
        ],
        "enc_ht_tea_list": [
            torch.randn(batch_size, 128, timesteps, 129, device=device),
            torch.randn(batch_size, 128, timesteps, 64, device=device),
            torch.randn(batch_size, 128, timesteps, 64, device=device),
        ],
        "dec_stu_fea_list": [
            torch.randn(batch_size, 64, timesteps, 64, device=device),
            torch.randn(batch_size, 64, timesteps, 129, device=device),
            torch.randn(batch_size, 2, timesteps, 257, device=device),
        ],
        "dec_tea_fea_list": [
            torch.randn(batch_size, 128, timesteps, 64, device=device),
            torch.randn(batch_size, 128, timesteps, 129, device=device),
            torch.randn(batch_size, 2, timesteps, 257, device=device),
        ],
    }


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    demo = TrainerKDStepDemo().to(device)
    demo_inputs = make_random_inputs(device)
    losses = demo(demo_inputs)

    for name, value in losses.items():
        print("{}: {:.6f}".format(name, value.item()))
