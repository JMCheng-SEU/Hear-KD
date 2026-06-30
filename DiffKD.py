import torch
from torch import nn
import torch.nn.functional as F


class DDIMScheduler:
    def __init__(
        self,
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        clip_sample=False,
        set_alpha_to_one=True,
        steps_offset=0,
        prediction_type="epsilon",
    ):
        if beta_schedule != "linear":
            raise NotImplementedError("This demo keeps only the linear beta schedule.")

        self.num_train_timesteps = num_train_timesteps
        self.clip_sample = clip_sample
        self.steps_offset = steps_offset
        self.prediction_type = prediction_type
        self.betas = torch.linspace(
            beta_start, beta_end, num_train_timesteps, dtype=torch.float32
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        if set_alpha_to_one:
            self.final_alpha_cumprod = torch.tensor(1.0)
        else:
            self.final_alpha_cumprod = self.alphas_cumprod[0]
        self.num_inference_steps = None
        self.timesteps = torch.arange(num_train_timesteps - 1, -1, -1).long()

    def set_timesteps(self, num_inference_steps, device=None):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = torch.arange(0, num_inference_steps, device=device)
        timesteps = (timesteps * step_ratio).round().long().flip(0)
        self.timesteps = timesteps + self.steps_offset

    def _get_variance(self, timestep, prev_timestep, device, dtype):
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=dtype)
        final_alpha_cumprod = self.final_alpha_cumprod.to(device=device, dtype=dtype)

        alpha_prod_t = alphas_cumprod[timestep]
        if prev_timestep >= 0:
            alpha_prod_t_prev = alphas_cumprod[prev_timestep]
        else:
            alpha_prod_t_prev = final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )
        return variance

    def step(
        self,
        model_output,
        timestep,
        sample,
        eta=0.0,
        use_clipped_model_output=False,
        generator=None,
        variance_noise=None,
    ):
        if self.num_inference_steps is None:
            raise ValueError("set_timesteps must be called before step.")

        timestep = int(timestep.item()) if torch.is_tensor(timestep) else int(timestep)
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps

        device = sample.device
        dtype = sample.dtype
        alphas_cumprod = self.alphas_cumprod.to(device=device, dtype=dtype)
        final_alpha_cumprod = self.final_alpha_cumprod.to(device=device, dtype=dtype)

        alpha_prod_t = alphas_cumprod[timestep]
        if prev_timestep >= 0:
            alpha_prod_t_prev = alphas_cumprod[prev_timestep]
        else:
            alpha_prod_t_prev = final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t.sqrt() * model_output
            ) / alpha_prod_t.sqrt()
        elif self.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                alpha_prod_t.sqrt() * sample - beta_prod_t.sqrt() * model_output
            )
            model_output = (
                alpha_prod_t.sqrt() * model_output + beta_prod_t.sqrt() * sample
            )
        else:
            raise ValueError("Unsupported prediction_type: {}".format(self.prediction_type))

        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        variance = self._get_variance(timestep, prev_timestep, device, dtype)
        std_dev_t = eta * variance.sqrt()

        if use_clipped_model_output:
            model_output = (
                sample - alpha_prod_t.sqrt() * pred_original_sample
            ) / beta_prod_t.sqrt()

        pred_sample_direction = (
            1 - alpha_prod_t_prev - std_dev_t ** 2
        ).sqrt() * model_output
        prev_sample = alpha_prod_t_prev.sqrt() * pred_original_sample + pred_sample_direction

        if eta > 0:
            if variance_noise is not None and generator is not None:
                raise ValueError("Only one of generator and variance_noise can be set.")
            if variance_noise is None:
                variance_noise = torch.randn(
                    model_output.shape,
                    generator=generator,
                    device=device,
                    dtype=dtype,
                )
            prev_sample = prev_sample + variance.sqrt() * eta * variance_noise

        return {"prev_sample": prev_sample, "pred_original_sample": pred_original_sample}

    def add_noise(self, original_samples, noise, timesteps):
        self.alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)

        sqrt_alpha_prod = self.alphas_cumprod[timesteps].sqrt().flatten()
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (
            1 - self.alphas_cumprod[timesteps]
        ).sqrt().flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        return sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise


class LearnableSigmoid(nn.Module):
    def __init__(self, in_features, beta=1):
        super(LearnableSigmoid, self).__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class NoiseAdapter_Ht(nn.Module):
    def __init__(self, ndf, in_channel=2):
        super(NoiseAdapter_Ht, self).__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel * 2, ndf, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf, ndf * 2, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.PReLU(2 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 2, ndf * 4, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * 4, ndf * 8, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.PReLU(8 * ndf),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf * 8, ndf * 4)),
            nn.Dropout(0.3),
            nn.PReLU(4 * ndf),
            nn.utils.spectral_norm(nn.Linear(ndf * 4, 1)),
            LearnableSigmoid(1),
        )

    def forward(self, feat, ht_emb):
        xy_ht = torch.cat([feat, ht_emb], dim=1)
        return self.layers(xy_ht)[:, 0]


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4):
        super(Bottleneck, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, 3, padding=1),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x) + x


class DiffusionModel(nn.Module):
    def __init__(self, channels_in, kernel_size=3):
        super(DiffusionModel, self).__init__()
        self.time_embedding = nn.Embedding(1280, channels_in)

        if kernel_size == 3:
            self.pred = nn.Sequential(
                Bottleneck(channels_in, channels_in),
                Bottleneck(channels_in, channels_in),
                nn.Conv2d(channels_in, channels_in, 1),
                nn.BatchNorm2d(channels_in),
            )
        else:
            self.pred = nn.Sequential(
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
                nn.BatchNorm2d(channels_in),
                nn.Conv2d(channels_in, channels_in * 4, 1),
                nn.BatchNorm2d(channels_in * 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels_in * 4, channels_in, 1),
            )

    def forward(self, noisy_image, t):
        if t.dtype != torch.long:
            t = t.type(torch.long)
        feat = noisy_image + self.time_embedding(t)[..., None, None]
        return self.pred(feat)


class AutoEncoder(nn.Module):
    def __init__(self, channels, latent_channels):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, latent_channels, 1, padding=0),
            nn.BatchNorm2d(latent_channels),
        )
        self.decoder = nn.Conv2d(latent_channels, channels, 1, padding=0)

    def forward(self, x):
        hidden = self.encoder(x)
        out = self.decoder(hidden)
        return hidden, out


class HtApapt_DDIMPipeline:
    def __init__(self, model, scheduler, noise_adapter_gaussian=None):
        self.model = model
        self.scheduler = scheduler
        self.noise_adapter_gaussian = noise_adapter_gaussian

    def __call__(
        self,
        batch_size,
        device,
        dtype,
        shape,
        feat,
        noise_feat,
        ht_feat,
        generator=None,
        eta=0.0,
        num_inference_steps=50,
        proj=None,
    ):
        image_shape = (batch_size, *shape)

        if self.noise_adapter_gaussian is not None:
            noise_gaussian = torch.randn(image_shape, device=device, dtype=dtype)
            gamma_gaussian = self.noise_adapter_gaussian(feat, ht_feat)
            while len(gamma_gaussian.shape) < len(noise_gaussian.shape):
                gamma_gaussian = gamma_gaussian.unsqueeze(-1)
            image = feat + gamma_gaussian * noise_gaussian
        else:
            image = feat

        self.scheduler.set_timesteps(num_inference_steps * 2, device=device)
        for t in self.scheduler.timesteps[len(self.scheduler.timesteps) // 2:]:
            noise_pred = self.model(image, t.to(device))
            image = self.scheduler.step(
                noise_pred,
                t,
                image,
                eta=eta,
                use_clipped_model_output=True,
                generator=generator,
            )["prev_sample"]

        return image


class DiffKD_Ht(nn.Module):
    def __init__(
        self,
        student_channels,
        teacher_channels,
        kernel_size=3,
        inference_steps=5,
        num_train_timesteps=1000,
        use_ae=False,
        ae_channels=None,
        F_shape=64,
        trans_stu=False,
    ):
        super(DiffKD_Ht, self).__init__()
        self.use_ae = use_ae
        self.diffusion_inference_steps = inference_steps
        self.trans_stu = trans_stu

        if use_ae:
            if ae_channels is None:
                ae_channels = teacher_channels // 2
            self.ae = AutoEncoder(teacher_channels, ae_channels)
            teacher_channels = ae_channels

        if self.trans_stu:
            self.trans = nn.Conv2d(student_channels, teacher_channels, 1)

        self.model = DiffusionModel(channels_in=teacher_channels, kernel_size=kernel_size)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            clip_sample=False,
            beta_schedule="linear",
        )
        self.noise_adapter_gaussian = NoiseAdapter_Ht(ndf=16, in_channel=teacher_channels)
        self.pipeline = HtApapt_DDIMPipeline(
            self.model,
            self.scheduler,
            noise_adapter_gaussian=self.noise_adapter_gaussian,
        )
        self.proj = nn.Sequential(
            nn.Conv2d(teacher_channels, teacher_channels, 1),
            nn.BatchNorm2d(teacher_channels),
        )

        self.comu_conv_ht = nn.Conv2d(1, teacher_channels, kernel_size=(1, 1))
        self.comu_linear_ht = nn.Linear(257, F_shape)
        self.comu_conv_noise = nn.Conv2d(2, teacher_channels, kernel_size=(1, 1))
        self.comu_linear_noise = nn.Linear(257, F_shape)

    def forward(self, student_feat, teacher_feat, noise_complex, ht_emb):
        if self.trans_stu:
            student_feat = self.trans(student_feat)

        noise_feat = self.comu_conv_noise(self.comu_linear_noise(noise_complex))
        ht_feat = self.comu_conv_ht(self.comu_linear_ht(ht_emb))
        teacher_feat = teacher_feat.detach()

        if self.use_ae:
            hidden_t_feat, rec_t_feat = self.ae(teacher_feat)
            rec_loss = F.mse_loss(teacher_feat, rec_t_feat)
            teacher_feat = hidden_t_feat.detach()
        else:
            rec_loss = None

        refined_feat = self.pipeline(
            batch_size=student_feat.shape[0],
            device=student_feat.device,
            dtype=student_feat.dtype,
            shape=student_feat.shape[1:],
            feat=student_feat,
            noise_feat=noise_feat,
            ht_feat=ht_feat,
            num_inference_steps=self.diffusion_inference_steps,
            proj=self.proj,
        )
        refined_feat = self.proj(refined_feat)

        ddim_loss = self.ddim_loss(teacher_feat)
        return refined_feat, teacher_feat, ddim_loss, rec_loss

    def ddim_loss(self, gt_feat):
        noise = torch.randn(gt_feat.shape, device=gt_feat.device)
        batch_size = gt_feat.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch_size,),
            device=gt_feat.device,
        ).long()
        noisy_images = self.scheduler.add_noise(gt_feat, noise, timesteps)
        noise_pred = self.model(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        return loss
