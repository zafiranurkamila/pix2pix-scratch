"""Fungsi loss Pix2Pix: adversarial (BCEWithLogits) + L1.

Sesuai KONTRAK.md bagian 1 dan 5:
- Discriminator mengeluarkan logits mentah [B, 1, 30, 30] (tanpa sigmoid),
  jadi adversarial loss memakai BCEWithLogitsLoss.
- Label real/fake dibuat dengan ones_like/zeros_like, tidak meng-hardcode 30x30.

File ini hanya berisi rumus loss. Memanggil G/D dan .detach() adalah tugas train.py.
"""

import torch
import torch.nn as nn
from torch import Tensor


class Pix2PixLoss:
    """Menghitung loss Discriminator dan Generator untuk Pix2Pix.

    Args:
        lambda_l1:  bobot L1 loss (paper Pix2Pix: 100).
        lambda_adv: bobot adversarial loss untuk G (default 1).
                    Set 0 untuk eksperimen ablation "L1 saja".
    """

    def __init__(self, lambda_l1: float = 100.0, lambda_adv: float = 1.0):
        if lambda_l1 < 0 or lambda_adv < 0:
            raise ValueError("lambda_l1 dan lambda_adv tidak boleh negatif")
        self.lambda_l1 = lambda_l1
        self.lambda_adv = lambda_adv
        self._bce = nn.BCEWithLogitsLoss()
        self._l1 = nn.L1Loss()

    def _adversarial(self, logits: Tensor, is_real: bool) -> Tensor:
        """BCE antara logits D dan label yang semuanya 1 (real) atau 0 (fake)."""
        label = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
        return self._bce(logits, label)

    def discriminator_loss(self, real_logits: Tensor, fake_logits: Tensor) -> tuple[Tensor, dict]:
        """Loss D: pasangan asli harus dinilai 1, pasangan palsu dinilai 0.

        real_logits = D(cond, target)
        fake_logits = D(cond, fake.detach())   <- detach dilakukan di train.py
        """
        loss_real = self._adversarial(real_logits, is_real=True)
        loss_fake = self._adversarial(fake_logits, is_real=False)
        # Dibagi 2 (sesuai paper) supaya D belajar lebih lambat relatif terhadap G.
        loss = 0.5 * (loss_real + loss_fake)
        parts = {
            "D_real": loss_real.item(),
            "D_fake": loss_fake.item(),
            "D_total": loss.item(),
        }
        return loss, parts

    def generator_loss(self, fake_logits: Tensor, fake: Tensor, target: Tensor) -> tuple[Tensor, dict]:
        """Loss G: menipu D (pasangan palsu dinilai 1) + mirip target secara piksel.

        fake_logits = D(cond, fake)   <- TANPA detach, supaya gradien sampai ke G
        """
        loss_adv = self._adversarial(fake_logits, is_real=True)
        loss_l1 = self._l1(fake, target)
        loss = self.lambda_adv * loss_adv + self.lambda_l1 * loss_l1
        parts = {
            "G_adv": loss_adv.item(),
            "G_L1": loss_l1.item(),
            "G_total": loss.item(),
        }
        return loss, parts


if __name__ == "__main__":
    # Smoke test: py -m src.losses
    import math
    from src.models.discriminator import PatchGANDiscriminator, init_weights

    torch.manual_seed(42)
    crit = Pix2PixLoss(lambda_l1=100.0)

    # 1) Nilai yang bisa dihitung manual
    zeros = torch.zeros(2, 1, 30, 30)
    assert math.isclose(crit._adversarial(zeros, True).item(), math.log(2), rel_tol=1e-6)
    a = torch.full((2, 3, 256, 256), -1.0)
    b = torch.full((2, 3, 256, 256), 1.0)
    assert math.isclose(crit._l1(a, b).item(), 2.0)
    print("Cek manual   : logits 0 -> BCE = ln 2, L1(-1, 1) = 2  OK")

    # 2) Alur seperti di training loop, dengan D asli dan 'fake' tiruan
    D = PatchGANDiscriminator()
    init_weights(D)
    cond = torch.rand(2, 3, 256, 256) * 2 - 1
    target = torch.rand(2, 3, 256, 256) * 2 - 1
    fake = (torch.rand(2, 3, 256, 256) * 2 - 1).requires_grad_(True)  # anggap output G

    loss_D, parts_D = crit.discriminator_loss(D(cond, target), D(cond, fake.detach()))
    loss_D.backward()
    assert fake.grad is None, "Loss D tidak boleh mengalirkan gradien ke G"
    print("Loss D       :", {k: round(v, 4) for k, v in parts_D.items()})

    D.zero_grad()
    loss_G, parts_G = crit.generator_loss(D(cond, fake), fake, target)
    loss_G.backward()
    assert fake.grad is not None, "Loss G harus mengalirkan gradien ke G"
    print("Loss G       :", {k: round(v, 4) for k, v in parts_G.items()})

    assert not any(math.isnan(v) for v in {**parts_D, **parts_G}.values()), "Ada NaN"
    print("Smoke test losses: LOLOS")
