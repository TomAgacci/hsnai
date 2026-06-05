def omega_phi_AL_step(model, inputs, targets, optimizer):
    optimizer.zero_grad()

    outputs = model(inputs)
    loss = my_loss_fn(outputs, targets)

    x_t = intensity_metric(outputs)  # scalar
    x_val = x_t.item()

    # deviation from golden band
    if x_val < L:
        d_val = L - x_val
    elif x_val > U:
        d_val = x_val - U
    else:
        d_val = 0.0

    d = torch.tensor(d_val, dtype=loss.dtype, device=loss.device)
    f_t = torch.exp(-alpha * d**2)  # Dawkins-style fitness

    # main loss scaled by golden-mean fitness
    (f_t * loss).backward(retain_graph=True)

    # Heideggerian correction if outside band
    if d_val > 0.0:
        optimizer.zero_grad()
        x_t.backward(retain_graph=True)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = lambda_ * (x_val - x_star) * p.grad
        optimizer.step()
    else:
        optimizer.step()

    return {"loss": loss.item(), "x_t": x_val, "d": d_val, "f_t": f_t.item()}
