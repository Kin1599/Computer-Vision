import statistics

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    # pin_memory ускоряет передачу CPU -> GPU
    # num_workers и persistent_workers позволяют DataLoader заранее готовить батчи
    dataloader = DataLoader(
        prepare_data(),
        batch_size=256, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    forward_times = []
    backward_times = []

    total_loss = torch.zeros((), device='cuda') # для накопления лоссов на GPU
    total_batches = 0

    # Создаем CUDA Event до цикла
    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)
    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        data = data.to('cuda', non_blocking=True)
        target = target.to('cuda', non_blocking=True)

        noise = torch.randn_like(data) # генерируем сразу на GPU
        data = data + noise
        
        # set_to_none=True быстрее и экономнее, чем занулять градиенты числом 0
        optimizer.zero_grad(set_to_none=True)

        start_fwd.record()
        output = model(data)
        loss = criterion(output, target)
        end_fwd.record()

        start_bwd.record()
        loss.backward()
        end_bwd.record()

        optimizer.step()

        batch_size = data.size(0)

        total_loss += loss.detach() * batch_size
        total_batches += batch_size

        torch.cuda.synchronize() # ждем завершения всех операций на GPU

        forward_times.append(start_fwd.elapsed_time(end_fwd) / 1000) # переводим в секунды
        backward_times.append(start_bwd.elapsed_time(end_bwd) / 1000)

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} loss: {loss.item():.4f}")

    avg_loss = (total_loss / total_batches).item()

    print(
        f"Epoch finished, avg_loss: {avg_loss:.4f}, "
        f"avg forward time is {statistics.mean(forward_times):.6f}, sec, "
        f"avg backward time is {statistics.mean(backward_times):.6f} sec"
    )

if __name__ == '__main__':
    train()