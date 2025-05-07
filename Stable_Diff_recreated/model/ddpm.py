"""
Ici, on crée le modèle qui va, a partir du bruit prédit par le unet, le retirer de l'image
DDPM = Denoising Diffusion Probabilistic Models
"""

import torch
import numpy as np

class DDPMSampler:
    def __init__(self, generator: torch.Generator, n_training_steps: int = 1000, beta_start: float = 0.00085, beta_end: float = 0.0120):
        """
        beta = variance du bruit ajouté à chaque étape de diffusion, On choisit une évolution linéaire de beta_start à beta_end
        """
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) #les alphas barre: [alpha_0, alpha_0 * alpha_1, ...]
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.n_training_steps = n_training_steps
        self.timesteps = torch.from_numpy(np.arange(0, n_training_steps)[::-1].copy())

    def set_inference_timesteps(self, n_inference_steps: int = 50):
        """
        On choisit le nombre d'étapes de diffusion à utiliser lors de l'inférence (n_training_steps n'est utilisé que pour l'entraînement, on fait moins
        d'itertions lors de l'inférence pour aller plus vite)
        """
        self.n_inference_steps = n_inference_steps
        #entrainement: 1000 steps: 999, 998, ..., 0
        #inference: 50 steps: 999, 979, ..., 0 (pas de 1000/50 = 20)
        delta_step = self.n_training_steps // n_inference_steps
        self.timesteps = torch.from_numpy((np.arange(0, n_inference_steps) * delta_step).round()[::-1].copy().astype(np.int64))


    def add_noise(self, original: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        """
        On ajoute du bruit à l'image en fonction de l'etape de diffusion (diffusion avant)
        On utilise la formule de diffusion pour ajouter le bruit:
        x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon
        """
        alphas_cumprod = self.alphas_cumprod.to(device = original.device, dtype = original.dtype)
        timesteps = timesteps.to(device = original.device)

        sqrt_alphas_cumprod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.flatten()
        #On ajoute des dimensions pour que sqrt_alphas_cumprod ait la même forme que original
        while len(sqrt_alphas_cumprod.shape) < len(original.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)

        mean = sqrt_alphas_cumprod * original

        #stdv
        sqrt_one_minus_alphas_cumprod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.flatten()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(original.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        stdv = sqrt_one_minus_alphas_cumprod

        noise = torch.randn(original.shape, generator = self.generator, device = original.device, dtype = original.dtype)

        noisified = mean + stdv * noise
        return noisified

    def step(self, timestep: int, x_t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        """
        Ici on retire le bruit (prédit par le unet) de l'image
        On utilise les formules du papier pour la diffusion arriere
        """
        t = timestep
        prev_t = t - (self.n_training_steps // self.n_inference_steps) 

        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_prev_t = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one

        beta_cumprod_t = 1 - alphas_cumprod_t
        beta_cumprod_prev_t = 1 - alphas_cumprod_prev_t

        alpha_t = alphas_cumprod_t / alphas_cumprod_prev_t
        beta_t = 1 - alpha_t

        #Calcul de la prediction de x0 (formule 15 du papier)
        pred_x0 = (x_t - beta_cumprod_t ** 0.5 * predicted_noise) / (alphas_cumprod_t ** 0.5)

        #Calcul de l'estimateur de la moyenne (formule 7 du papier)
        pred_mean = (alphas_cumprod_prev_t ** 0.5 * beta_t) / beta_cumprod_t * pred_x0 + \
                (alpha_t ** 0.5 * beta_cumprod_prev_t) / beta_cumprod_t * x_t

        #Calcul de l'estimateur de la variance (formule 7 du papier)
        pred_variance = 0
        if t > 0:
            device = predicted_noise.device
            noise = torch.randn(predicted_noise.shape, generator=self.generator, device=device, dtype=predicted_noise.dtype)
            pred_variance = (1 - alphas_cumprod_prev_t) / (1 - alphas_cumprod_t) * beta_t
            pred_variance = torch.clamp(pred_variance, min = 1e-20)
            pred_variance = (pred_variance ** 0.5) * noise

        #Estimation de x_t-1 (formule 6 du papier)
        pred_x_tm1 = pred_mean + pred_variance

        return pred_x_tm1
    
    def set_strength(self, strength = 1):
        """
        Pour la diffusion image a image, cette methode permet de choisir la force de la diffusion
        (ie la quantité de bruit à ajouter à l'image d'entrée) avant le processus de denoising
        """
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step