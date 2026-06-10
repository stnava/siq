import os
import pytest
import siq
import ants
import numpy as np

def test_blind_sr_kitchen_sink():
    # 1. Setup a tiny cache for testing
    hr_large_shape = (48, 48, 48)
    cache = [siq.simulate_image_multi_scale(hr_large_shape) for _ in range(2)]
    
    # 2. Run kitchen-sink training for a few iterations
    # We'll use use_residual=False for a smaller model in tests
    model = siq.train_blind_sr_kitchen_sink(
        output_prefix="test_kitchen_sink",
        factor=2,
        iterations=5,
        hr_base_cache=cache,
        use_residual=False
    )
    
    assert model is not None
    assert os.path.exists("test_kitchen_sink_best.keras")
    
    # Clean up
    if os.path.exists("test_kitchen_sink_best.keras"):
        os.remove("test_kitchen_sink_best.keras")

def test_parameterized_simulation():
    hr_large_shape = (48, 48, 48)
    # Test with extreme parameters to ensure they are used
    sim_params = {
        'scale_range': (1.0, 1.0),
        'n_levels_range': (10, 11),
        'sigma_range': (0.5, 0.5)
    }
    img = siq.simulate_image_multi_scale(hr_large_shape, **sim_params)
    assert img.shape == hr_large_shape

    # Test generator with custom params
    gen = siq.blind_sr_generator(
        batch_size=1, 
        gamma_range=(1.0, 1.0), 
        noise_std_range=(0.1, 0.1),
        sim_params=sim_params
    )
    x, y = next(gen)
    assert x.shape == (1, 16, 16, 16, 1)

def test_custom_distribution():
    hr_large_shape = (48, 48, 48)
    # Poisson distribution for sigma_range will produce many zeros if lam is small
    sim_params = {
        'sigma_range': {'type': 'poisson', 'lam': 0.1}
    }
    img = siq.simulate_image_multi_scale(hr_large_shape, **sim_params)
    assert img.shape == hr_large_shape

    # Test with custom callable
    gen = siq.blind_sr_generator(
        batch_size=1, 
        blur_sigma_range=lambda: np.random.poisson(0.5)
    )
    x, y = next(gen)
    assert x.shape == (1, 16, 16, 16, 1)

if __name__ == "__main__":
    test_custom_distribution()
    test_parameterized_simulation()
    test_blind_sr_kitchen_sink()
