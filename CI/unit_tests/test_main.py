# from UI import main as o
from UI.main import MLP, ActorCritic

# Instantiate model
num_actions = 2  # 2
num_hidden_units = 128

model = ActorCritic(num_actions = num_actions,num_hidden_units = num_hidden_units)
print(model)
