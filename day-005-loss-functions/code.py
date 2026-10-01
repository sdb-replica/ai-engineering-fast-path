"""Day 5 — Loss Functions: How a Model Knows It's Wrong.

One concept: a loss function turns "how wrong was I?" into a single number
that training spends down. We compute MSE and MAE on a toy prediction,
watch one outlier punish MSE ~74x harder, and show the loss carries a
gradient — the downhill direction training will follow tomorrow.

Env: ~/ai-lab venv on the MacBook, PyTorch with MPS (falls back to CPU).
Run:  python3 code.py
"""
import torch


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Running on: {device}")

    # Toy setup: 4 hourly closes (scaled $), our model's prediction of each.
    target = torch.tensor([2.0, 3.1, 3.9, 5.2], device=device)
    pred = torch.tensor([2.2, 2.8, 4.0, 4.9], device=device)

    residual = pred - target  # prediction - target
    mse = torch.mean(residual ** 2)
    mae = torch.mean(torch.abs(residual))

    print("residuals:", [f"{v:.2f}" for v in residual.cpu().tolist()])
    print(f"MSE (mean): {mse.item():.4f}   | sum: {torch.sum(residual ** 2).item():.4f}")
    print(f"MAE (mean): {mae.item():.4f}   | sum: {torch.sum(torch.abs(residual)).item():.4f}")

    # One outlier: the 4th print spikes to 9.0 (flash print / bad tick).
    target_outlier = torch.tensor([2.0, 3.1, 3.9, 9.0], device=device)
    r2 = pred - target_outlier
    mse2 = torch.mean(r2 ** 2)
    mae2 = torch.mean(torch.abs(r2))

    print("--- one outlier: target[3] spikes to 9.0 ---")
    print("outlier residuals:", [f"{v:.2f}" for v in r2.cpu().tolist()])
    print(f"MSE: {mse2.item():.4f}   MAE: {mae2.item():.4f}")
    print(f"One outlier made MSE {(mse2 / mse).item():.1f}x bigger, "
          f"MAE only {(mae2 / mae).item():.1f}x bigger.")

    # Why training can use a loss: it's smooth, so it has a slope.
    pred_g = pred.clone().requires_grad_(True)
    loss = torch.mean((pred_g - target) ** 2)
    loss.backward()
    print("d(loss)/d(pred):", [f"{v:.2f}" for v in pred_g.grad.cpu().tolist()])
    print("The loss has a gradient -> it points downhill. Tomorrow we follow it.")


if __name__ == "__main__":
    main()
