model = A2C(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=7e-4,
    gamma=0.99,
    device=device,
)

print("A2C model created.")
print("Model device:", model.device)