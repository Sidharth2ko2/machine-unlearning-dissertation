"""
β-VAE for Knowledge Disentanglement (Stage 1 of DKF).

Architecture:
  - SharedEncoder  Q_φ : image → S (shared attributes across classes)
  - UniqueEncoder  Q_ψ : image → U (class-specific unique attributes)
  - Decoder        P_θ : (S, U) → reconstructed image
  - ClassifierO    θ_o : S → original class output  (supervises shared encoder)
  - ClassifierY    θ_y : U → retain class label      (supervises unique encoder)

Key idea (from paper Section 4.2):
  S captures features shared between forget and retain classes (e.g. "wings"
  shared by airplane and bird). This shared knowledge K is the confounder.
  U captures features unique to each class (e.g. "landing gear" for airplane).

  By separating S and U, we can generate counterfactual X_c = Decoder(S_f, U_r):
  a synthetic sample that has the forget class's SHARED features but the retain
  class's UNIQUE features — placed near the decision boundary.

Loss (Equation 9 in paper — weighted ELBO):
  L_U = -E[log P(O|S)]          # shared encoder predicts original output
      - E[log P(Y|U)]           # unique encoder predicts retain label
      - E[log P(X|S,U)]         # reconstruction quality
      + β * KL(Q_φ(S|X)||P(S))  # disentangle S from class-specific info
      + β * KL(Q_ψ(U|X)||P(U))  # keep U as class-specific
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BETA, LATENT_DIM, NUM_CLASSES


# ── Encoder ────────────────────────────────────────────────────────────────────

class ConvEncoder(nn.Module):
    """
    Convolutional encoder for 32×32 CIFAR images.
    Outputs (mu, logvar) for the reparameterization trick.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,   32,  4, stride=2, padding=1),  # 16×16
            nn.LeakyReLU(0.2),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),  # 8×8
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),  # 4×4
            nn.LeakyReLU(0.2),
            nn.Flatten(),                                  # 128*4*4 = 2048
        )
        self.fc_mu     = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)

    def forward(self, x):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


# ── Decoder ────────────────────────────────────────────────────────────────────

class ConvDecoder(nn.Module):
    """
    Convolutional decoder from latent (S, U) back to 32×32 image.
    Input: concatenation of S and U → [B, latent_dim_s + latent_dim_u]
    Output: reconstructed image [B, 3, 32, 32]
    """
    def __init__(self, latent_dim_s, latent_dim_u):
        super().__init__()
        self.fc = nn.Linear(latent_dim_s + latent_dim_u, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8×8
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32, 4, stride=2, padding=1),  # 16×16
            nn.ReLU(),
            nn.ConvTranspose2d(32,   3, 4, stride=2, padding=1),  # 32×32
            nn.Tanh(),
        )

    def forward(self, s, u):
        h = self.fc(torch.cat([s, u], dim=1))
        h = h.view(h.size(0), 128, 4, 4)
        return self.net(h)


# ── Full β-VAE ─────────────────────────────────────────────────────────────────

class BetaVAE(nn.Module):
    def __init__(self,
                 latent_dim_s = LATENT_DIM,
                 latent_dim_u = LATENT_DIM,
                 num_classes  = NUM_CLASSES,
                 beta         = BETA):
        super().__init__()
        self.beta = beta

        # Two separate encoders — same architecture, independent weights
        self.encoder_s = ConvEncoder(latent_dim_s)   # shared features
        self.encoder_u = ConvEncoder(latent_dim_u)   # unique features
        self.decoder   = ConvDecoder(latent_dim_s, latent_dim_u)

        # Classifiers that supervise the disentanglement
        self.classifier_o = nn.Linear(latent_dim_s, num_classes)  # S → original output
        self.classifier_y = nn.Linear(latent_dim_u, num_classes)  # U → retain label

    @staticmethod
    def reparameterize(mu, logvar):
        """Sample z ~ N(mu, sigma²) using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    @staticmethod
    def kl_divergence(mu, logvar):
        """KL(N(mu, sigma²) || N(0,1))."""
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def encode_shared(self, x):
        mu, logvar = self.encoder_s(x)
        return self.reparameterize(mu, logvar), mu, logvar

    def encode_unique(self, x):
        mu, logvar = self.encoder_u(x)
        return self.reparameterize(mu, logvar), mu, logvar

    def forward(self, x_f, x_r):
        """
        x_f: forget samples (airplane images)
        x_r: retain samples (non-airplane images)

        Returns:
          x_recon : reconstruction of x_f (using S_f + U_f)
          x_cf    : counterfactual (using S_f + U_r) ← key output
          pred_o  : classifier prediction from shared S_f
          pred_y  : classifier prediction from unique U_r
          kl_s, kl_u : KL divergence terms for the loss
        """
        # Encode shared features from forget samples
        s_f, mu_s, logvar_s = self.encode_shared(x_f)

        # Encode unique features from retain samples
        u_r, mu_u, logvar_u = self.encode_unique(x_r)

        # Encode unique features from forget samples (for reconstruction)
        u_f, _, _ = self.encode_unique(x_f)

        # Reconstruct forget image (verifies encoder-decoder quality)
        x_recon = self.decoder(s_f, u_f)

        # Generate counterfactual: forget's shared attributes + retain's unique attributes
        # X_c = Decoder(S_f, U_r) — placed near the decision boundary
        x_cf = self.decoder(s_f, u_r)

        # Supervised classification (enforces disentanglement)
        pred_o = self.classifier_o(s_f)  # S should encode class-agnostic shared info
        pred_y = self.classifier_y(u_r)  # U should encode class-specific info

        kl_s = self.kl_divergence(mu_s, logvar_s)
        kl_u = self.kl_divergence(mu_u, logvar_u)

        return x_recon, x_cf, pred_o, pred_y, kl_s, kl_u, s_f, u_r

    def compute_loss(self, x_f, x_r, y_r, y_f):
        """
        Compute β-VAE ELBO loss for training the disentanglement.

        Returns:
          total_loss : scalar loss for optimizer
          x_cf       : counterfactual image (used in DKF training)
        """
        x_recon, x_cf, pred_o, pred_y, kl_s, kl_u, s_f, u_r = self(x_f, x_r)

        # Reconstruction quality
        recon_loss = F.mse_loss(x_recon, x_f)

        # Classification supervision (enforces what S and U encode)
        cls_o = F.cross_entropy(pred_o, y_f)  # S_f should predict forget class
        cls_y = F.cross_entropy(pred_y, y_r)  # U_r should predict retain class

        # β-KL for disentanglement
        kl_loss = self.beta * (kl_s + kl_u)

        total = recon_loss + cls_o + cls_y + kl_loss
        return total, x_cf.detach()   # detach x_cf: VAE already updated, student treats it as fixed input
