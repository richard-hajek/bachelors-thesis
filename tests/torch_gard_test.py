import torch

# Creating the graph
x = torch.tensor([1.0, 1.5], requires_grad=True)
y = torch.tensor([2.0, 2.0], requires_grad=True)
z = x * y

# Displaying
z.backward(torch.tensor([1., 0.]))
for i, name in zip([x, y, z], "xyz"):
    print(f"{name} data: {i.data}")
    print(f"requires_grad: {i.requires_grad}")
    print(f"grad: {i.grad}")
    print(f"grad_fn: {i.grad_fn}")
    print(f"is_leaf: {i.is_leaf}\n")

print(x.grad.data)
