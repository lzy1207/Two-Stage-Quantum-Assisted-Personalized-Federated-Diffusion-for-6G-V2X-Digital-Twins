import unittest
import torch
from torch import nn

from qv2x.diffusion import ConditionalUNet, GaussianDiffusion, make_spatial_condition


class DiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(5)
        self.model = ConditionalUNet(1, 6, base_channels=8, channel_mults=(1, 2), embedding_dim=16)
        self.x = torch.randn(2, 1, 8, 8).clamp(-1, 1)
        self.prototype = torch.randn(2, 6)
        self.spatial = make_spatial_condition(self.x, torch.tensor([[.3, .7], [.2, .4]]), torch.zeros_like(self.x))

    def test_condition_location_and_shape(self):
        coarse = torch.zeros(1, 1, 5, 5)
        result = make_spatial_condition(coarse, torch.tensor([[.25, .75]]), None)
        self.assertEqual(result.shape, (1, 2, 5, 5))
        self.assertEqual(result[0, 1].argmax().item(), 3 * 5 + 1)
        self.assertEqual(result[0, 1, 3, 1].item(), 1)

    def test_q_sample_inverts_known_noise(self):
        diffusion = GaussianDiffusion(9)
        t = torch.tensor([0, 8])
        noise = torch.randn_like(self.x)
        noisy = diffusion.q_sample(self.x, t, noise)
        abar = diffusion.alpha_bars[t][:, None, None, None]
        recovered = (noisy - (1 - abar).sqrt() * noise) / abar.sqrt()
        torch.testing.assert_close(recovered, self.x)
        self.assertEqual(diffusion.posterior_variance[0].item(), 0)

    def test_training_gradients_reach_film_and_inputs(self):
        spatial = self.spatial.detach().requires_grad_(True)
        prototype = self.prototype.detach().requires_grad_(True)
        losses = GaussianDiffusion(10).training_loss(self.model, self.x, spatial, prototype)
        self.assertEqual(set(losses), {"loss", "noise", "rec", "grad", "ssim"})
        self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
        losses["loss"].backward()
        self.assertGreater(spatial.grad.abs().sum().item(), 0)
        self.assertGreater(prototype.grad.abs().sum().item(), 0)
        film_gradients = [parameter.grad for name, parameter in self.model.named_parameters() if ".film." in name]
        self.assertTrue(all(g is not None and g.abs().sum() > 0 for g in film_gradients))

    def test_sampling_determinism_and_mode_restore(self):
        diffusion = GaussianDiffusion(4)
        self.model.train()
        for steps in (None, 2):
            first = diffusion.sample(self.model, self.spatial, self.prototype, steps, torch.Generator().manual_seed(8))
            second = diffusion.sample(self.model, self.spatial, self.prototype, steps, torch.Generator().manual_seed(8))
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            self.assertEqual(first.shape, self.x.shape)
            self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(self.model.training)

    def test_last_ddpm_step_matches_closed_form_without_extra_noise(self):
        class ZeroModel(nn.Module):
            def forward(self, noisy, t, spatial, prototype):
                return torch.zeros_like(noisy)
        diffusion = GaussianDiffusion(1, .01, .01)
        initial = torch.randn(self.x.shape, generator=torch.Generator().manual_seed(12))
        actual = diffusion.sample(ZeroModel(), self.spatial, self.prototype, generator=torch.Generator().manual_seed(12))
        torch.testing.assert_close(actual, initial / (.99 ** .5))

    def test_checkpoint_model_identity(self):
        copy = ConditionalUNet(**self.model.config)
        copy.load_state_dict(self.model.state_dict(), strict=True)
        self.model.eval()
        copy.eval()
        t = torch.tensor([1, 2])
        torch.testing.assert_close(self.model(self.x, t, self.spatial, self.prototype),
                                   copy(self.x, t, self.spatial, self.prototype), rtol=0, atol=0)

    def test_masked_loss_and_invalid_inputs(self):
        diffusion = GaussianDiffusion(4)
        mask = torch.ones_like(self.x)
        mask[..., :2, :2] = 0
        losses = diffusion.training_loss(self.model, self.x, self.spatial, self.prototype, mask=mask)
        self.assertTrue(torch.isfinite(losses["loss"]))
        with self.assertRaises(ValueError):
            diffusion.training_loss(self.model, self.x, self.spatial, self.prototype, mask=torch.zeros_like(mask))
        with self.assertRaises(ValueError):
            diffusion.sample(self.model, self.spatial, self.prototype, steps=5)

    def test_structural_loss_respects_normalized_data_range(self):
        diffusion = GaussianDiffusion(4)
        torch.manual_seed(15)
        narrow = diffusion.training_loss(self.model, self.x, self.spatial, self.prototype, data_range=2)
        torch.manual_seed(15)
        wide = diffusion.training_loss(self.model, self.x, self.spatial, self.prototype, data_range=20)
        torch.testing.assert_close(narrow["noise"], wide["noise"], rtol=0, atol=0)
        self.assertNotEqual(narrow["ssim"].item(), wide["ssim"].item())
        with self.assertRaises(ValueError):
            diffusion.training_loss(self.model, self.x, self.spatial, self.prototype, data_range=0)


if __name__ == "__main__":
    unittest.main()
