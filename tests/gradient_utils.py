"""Utilities for gradient testing."""

import torch


def sampled_gradcheck(
    fn,
    inputs: tuple[torch.Tensor, ...],
    *,
    n_samples: int = 32,
    eps: float = 1e-6,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    seed: int = 42,
) -> bool:
    """Fast gradient check by sampling random input/output dimensions.

    Instead of checking all input-output pairs (which is O(n_inputs * n_outputs)),
    this samples random dimensions to check, making it O(n_samples).

    Args:
        fn: Function to check gradients for.
        inputs: Tuple of input tensors (must have requires_grad=True for those to check).
        n_samples: Number of random dimensions to sample per input tensor.
        eps: Epsilon for numerical gradient computation.
        atol: Absolute tolerance for gradient comparison.
        rtol: Relative tolerance for gradient comparison.
        seed: Random seed for reproducibility.

    Returns:
        True if gradients match, raises AssertionError otherwise.
    """
    torch.manual_seed(seed)

    for inp_idx, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue

        # Ensure we can get gradients for this tensor
        if not inp.is_leaf:
            inp.retain_grad()

        n_inputs = inp.numel()

        # Get output size from a forward pass
        with torch.no_grad():
            output = fn(*inputs)
            if isinstance(output, tuple):
                output = output[0]
            n_outputs = output.numel()

        # Sample random input/output pairs
        sampled_in = torch.randint(0, n_inputs, (n_samples,))
        sampled_out = torch.randint(0, n_outputs, (n_samples,))

        for i in range(n_samples):
            in_idx_flat = sampled_in[i].item()
            out_idx_flat = sampled_out[i].item()

            # Zero all gradients
            for t in inputs:
                if t.requires_grad and t.grad is not None:
                    t.grad.zero_()

            # Compute analytical gradient via backprop
            output = fn(*inputs)
            if isinstance(output, tuple):
                output = output[0]

            # Create gradient tensor for selected output
            # Use contiguous + view(-1) instead of flatten() to ensure assignment persists
            grad_output = torch.zeros_like(output).contiguous()
            grad_output.view(-1)[out_idx_flat] = 1.0
            output.backward(grad_output)

            if inp.grad is None:
                analytical = 0.0
            else:
                analytical = inp.grad.flatten()[in_idx_flat].item()

            # Compute numerical gradient via finite differences
            orig_val = inp.flatten()[in_idx_flat].item()

            # f(x + eps)
            with torch.no_grad():
                inp.flatten()[in_idx_flat] = orig_val + eps
            with torch.no_grad():
                out_plus = fn(*inputs)
                if isinstance(out_plus, tuple):
                    out_plus = out_plus[0]
                f_plus = out_plus.flatten()[out_idx_flat].item()

            # f(x - eps)
            with torch.no_grad():
                inp.flatten()[in_idx_flat] = orig_val - eps
            with torch.no_grad():
                out_minus = fn(*inputs)
                if isinstance(out_minus, tuple):
                    out_minus = out_minus[0]
                f_minus = out_minus.flatten()[out_idx_flat].item()

            # Restore original value
            with torch.no_grad():
                inp.flatten()[in_idx_flat] = orig_val

            numerical = (f_plus - f_minus) / (2 * eps)

            # Compare - skip if both gradients are tiny (numerical noise)
            max_grad = max(abs(analytical), abs(numerical))
            if max_grad < 1e-3:
                continue  # Both gradients near zero, skip

            diff = abs(analytical - numerical)
            tol = atol + rtol * max_grad

            if diff > tol:
                raise AssertionError(
                    f"Gradient mismatch for input {inp_idx}, "
                    f"in_dim {in_idx_flat}, out_dim {out_idx_flat}: "
                    f"analytical={analytical:.6e}, numerical={numerical:.6e}, "
                    f"diff={diff:.6e}, tol={tol:.6e}"
                )

    return True


def prepare_params(params: dict[str, torch.Tensor], noise_scale: float = 0.01) -> dict[str, torch.Tensor]:
    """Prepare params for gradient checking: convert to float64, add noise, enable gradients.

    Creates leaf tensors with requires_grad=True to ensure gradients are computed.
    """
    result = {}
    for k, v in params.items():
        if v.dtype.is_floating_point:
            # Create a new leaf tensor with noise
            v = (
                (v.to(torch.float64) + torch.randn_like(v.to(torch.float64)) * noise_scale)
                .detach()
                .requires_grad_(True)
            )
        result[k] = v
    return result
