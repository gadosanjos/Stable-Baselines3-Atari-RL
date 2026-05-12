device = "cuda" if torch.cuda.is_available() else "cpu"

model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1_000,
    exploration_fraction=0.10,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.01,
    device=device,
)

print("DQN model created.")
print("Model device:", model.device)