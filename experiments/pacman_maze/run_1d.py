from argparse import ArgumentParser
import random
import cv2
import wandb
import torch
import sys
import time
from torch import nn
from torch import optim
from tqdm import tqdm
import scallopy
from collections import namedtuple, deque
import os

from arena import AvoidingArena, crop_cell_image_torch

FILE_DIR = os.path.abspath(os.path.join(os.path.abspath(__file__), "../"))

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

class CellClassifier(nn.Module):
  """
  Classifies each cell (in image format) into one of 4 classes: agent, goal, enemy, [empty]
  """
  def __init__(self):
    super(CellClassifier, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4, padding=2)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=4, padding=1)
    self.fc1 = nn.Linear(in_features=288, out_features=256)
    self.fc2 = nn.Linear(in_features=256, out_features=4)
    self.relu = nn.ReLU()

  def forward(self, x):
    batch_size, _, _, _ = x.shape
    x = self.relu(self.conv1(x)) # In: (80, 80, 4) Out: (20, 20, 16)
    x = self.relu(self.conv2(x)) # In: (20, 20, 16) Out: (10, 10, 32)
    x = x.view(batch_size, -1)
    x = self.relu(self.fc1(x)) # In: (3200,) Out: (256,)
    x = self.fc2(x) # In: (256,) Out: (4,)
    x = torch.softmax(x, dim=1)
    return x


class EntityExtractor(nn.Module):
  """
  Divide the whole image into grid cells, and pass the grid cells into the CellFeatureNet.
  The output of this network is 3 separate vectors: is_actor, is_goal, is_enemy,
  Each vector is of length #cells, mapping each cell to respective property (actor, goal, enemy)
  """
  def __init__(self, grid_x, grid_y, cell_pixel_size):
    super(EntityExtractor, self).__init__()
    self.grid_x = grid_x
    self.grid_y = grid_y
    self.cell_pixel_size = cell_pixel_size
    self.cell_dim = (self.grid_x, self.grid_y)
    self.cells = [(i, j) for i in range(grid_x) for j in range(grid_y)]
    self.cell_feature_net = CellClassifier()

  def forward(self, x):
    batch_size, _, _, _ = x.shape
    num_cells = len(self.cells)
    cells = torch.stack([torch.stack([crop_cell_image_torch(x[i], self.cell_dim, self.cell_pixel_size, c) for c in self.cells]) for i in range(batch_size)])
    cells = cells.reshape(batch_size * num_cells, 3, self.cell_pixel_size[0], self.cell_pixel_size[1])
    features = self.cell_feature_net(cells)
    batched_features = features.reshape(batch_size, num_cells, 4)
    is_actor = batched_features[:, :, 0]
    is_goal = batched_features[:, :, 1]
    is_enemy = batched_features[:, :, 2]
    return (is_actor, is_goal, is_enemy)


class PolicyNet(nn.Module):
  """
  A policy net that takes in an image and return the action scores as [UP, RIGHT, BOTTOM, LEFT]
  """
  def __init__(self, grid_x, grid_y, cell_pixel_size):
    super(PolicyNet, self).__init__()
    self.grid_x, self.grid_y = grid_x, grid_y
    self.cells = [(p,) for p in range(grid_x * grid_y)]

    # Setup CNNs that process the image and extract features
    self.extract_entity = EntityExtractor(grid_x, grid_y, cell_pixel_size)

    up_adjacency = []
    right_adjacency = []
    down_adjacency = []
    left_adjacency = []
    for p1 in range(grid_x*grid_y):
      for p2 in range(grid_x*grid_y):
        if p1 - grid_x == p2:
          up_adjacency.append((p1, p2))
        if p1 + grid_x == p2:
          down_adjacency.append((p1, p2))
        if p1 - 1 == p2 and p1 % grid_x != 0:
          left_adjacency.append((p1, p2))
        if p1 + 1 == p2 and p1 % grid_x != grid_x - 1:
          right_adjacency.append((p1, p2))
    assert(len(up_adjacency) == grid_x * (grid_y - 1))
    assert(len(right_adjacency) == (grid_x - 1) * grid_y)
    assert(len(down_adjacency) == grid_x * (grid_y - 1))
    assert(len(left_adjacency) == (grid_x - 1) * grid_y)

    # Setup scallop context and scallop forward functions
    self.path_planner = scallopy.Module(
      program="""
        // Static input facts
        type grid_node(p: u32)

        // Input from neural networks
        type actor(p: u32)
        type goal(p: u32)
        type enemy(p: u32)

        // Possible actions to take
        //type Action = Up | Right | Down | Left

        type up_adjacency(p1: u32, p2: u32)
        type right_adjacency(p1: u32, p2: u32)
        type down_adjacency(p1: u32, p2: u32)
        type left_adjacency(p1: u32, p2: u32)

        type edge_(p1: u32, p2: u32, a: u32)

        // Basic connectivity
        rel node_(p ) = grid_node(p), not enemy(p)
        rel edge_(p1, p2, 0) = node_(p1), node_(p2), up_adjacency(p1,p2)    // Up
        rel edge_(p1, p2, 1) = node_(p1), node_(p2), right_adjacency(p1,p2) // Right
        rel edge_(p1, p2, 2) = node_(p1), node_(p2), down_adjacency(p1,p2)  // Down
        rel edge_(p1, p2, 3) = node_(p1), node_(p2), left_adjacency(p1,p2)  // Left

        // Path for connectivity; will condition on no enemy on the path
        rel path(p, p) = node_(p)
        rel path(p1, p2) = edge_(p1, p2, _)
        rel path(p1, p3) = path(p1, p2), edge_(p2, p3, _)

        // Get the next position
        rel next_position(a, p2 ) = actor(p1 ), edge_(p1, p2, a)
        type next_action_tmp(x: u32, y: u32)
        rel next_action_tmp(a, 1) = next_position(a, p ), goal(pg ), path(p, pg)
        rel next_action(a) = next_action_tmp(a, _)

        // Constraint violation
        rel too_many_goal() = n := count(p: goal(p)), n > 1
        rel too_many_enemy() = n := count(p: enemy(p)), n > 5
        rel violation() = too_many_goal() or too_many_enemy()
      """.format(grid_width=grid_x),
      #provenance="diffminmaxprob",
      #provenance="diffaddmultprob",
      provenance="difftopkproofs",
      k=1,
      non_probabilistic=["up_adjacency", "right_adjacency", "down_adjacency", "left_adjacency"],
      facts={"grid_node": [(torch.tensor(args.attenuation, requires_grad=False), c) for c in self.cells],
             "up_adjacency": up_adjacency,
             "right_adjacency": right_adjacency,
             "down_adjacency": down_adjacency,
             "left_adjacency": left_adjacency
             },
      input_mappings={"actor": self.cells, "goal": self.cells, "enemy": self.cells},
      retain_topk={"actor": 3, "goal": 3, "enemy": 7},
      output_mappings={"next_action": list(range(4)), "violation": ()}, dispatch='single')

  def forward(self, x):
    actor, goal, enemy = self.extract_entity(x)
    result = self.path_planner(actor=actor, goal=goal, enemy=enemy, outputs = ["next_action", "violation"])
    next_action = torch.softmax(result["next_action"], dim=1)
    violation = result["violation"] * 0.2
    return next_action, violation

  def visualize(self, arena, raw_image, torch_image):
    curr_position, goal_position, is_enemy = self.extract_entity(torch_image)
    for (i, c) in enumerate(self.cells):
      blue, green, red = curr_position[0, i], goal_position[0, i], is_enemy[0, i]
      arena.paint_color(raw_image, (blue, green, red), c)
    return


class ReplayMemory:
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, transition):
    self.memory.append(transition)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)


class DQN:
  def __init__(self, grid_x, grid_y, cell_pixel_size):
    self.policy_net = PolicyNet(grid_x, grid_y, cell_pixel_size)

    # Create another network
    self.target_net = PolicyNet(grid_x, grid_y, cell_pixel_size)
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    # Store replay memory
    self.memory = ReplayMemory(args.replay_memory_capacity)

    # Loss function and optimizer
    self.criterion = nn.HuberLoss()
    self.violation_loss = nn.SmoothL1Loss()
    self.optimizer = optim.RMSprop(self.policy_net.parameters(), args.learning_rate)

  def predict_action(self, state_image):
    action_scores, _ = self.policy_net(state_image) # [0.25, 0.24, 0.26, 0.25]
    action = torch.argmax(action_scores, dim=1) # 2
    return action

  def observe_transition(self, transition):
    self.memory.push(transition)

  def optimize_model(self):
    if len(self.memory) < args.batch_size: return 0.0

    # Pull out a batch and its relevant features
    batch = self.memory.sample(args.batch_size)
    non_final_mask = torch.tensor([transition.next_state != None for transition in batch], dtype=torch.bool)
    non_final_next_states = torch.stack([transition.next_state[0] for transition in batch if transition.next_state is not None])
    action_batch = torch.stack([transition.action for transition in batch])
    state_batch = torch.stack([transition.state[0] for transition in batch])
    reward_batch = torch.stack([torch.tensor(transition.reward) for transition in batch])

    # Prepare the loss function
    state_action_values_raw, violations = self.policy_net(state_batch)
    state_action_values = state_action_values_raw.gather(1, action_batch)[:, 0]
    next_state_values = torch.zeros(args.batch_size)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states)[0].max(1)[0].detach()
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute the loss:
    loss1 = self.criterion(state_action_values, expected_state_action_values) # loss1 is the criterion
    loss2 = self.violation_loss(violations, torch.zeros(args.batch_size)) # loss2 is the violation
    loss = loss1 + loss2

    self.optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in self.policy_net.parameters():
      param.grad.data.clamp_(-1, 1)
    self.optimizer.step()

    # Return loss
    return loss.detach()

  def update_target(self):
    self.target_net.load_state_dict(self.policy_net.state_dict())


class Trainer:
  def __init__(self, grid_x, grid_y, cell_size, dpi, num_enemies, epsilon, num_epochs, args):
    self.args = args
    self.num_epochs = num_epochs
    self.arena = AvoidingArena((grid_x, grid_y), cell_size, dpi, num_enemies, easy=self.args.easy)
    self.dqn = DQN(grid_x, grid_y, self.arena.cell_pixel_size())
    self.epsilon = epsilon

  def show_image(self, raw_image, torch_image):
    if args.overlay_prediction:
      self.dqn.policy_net.visualize(self.arena, raw_image, torch_image)
    cv2.namedWindow("Current Arena", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Current Arena", args.window_size, args.window_size)
    cv2.imshow("Current Arena", raw_image)
    cv2.waitKey(int(args.show_run_interval * 1000))

  def train_epoch(self, epoch):
    self.epsilon = self.epsilon * self.args.epsilon_falloff
    success, failure, optimize_count, sum_loss = 0, 0, 0, 0.0
    iterator = tqdm(range(self.args.num_train_episodes))
    for episode_i in iterator:
      _ = self.arena.reset()
      curr_raw_image = self.arena.render()
      curr_state_image = self.arena.render_torch_tensor(image=curr_raw_image)
      for i in range(self.args.num_steps):
        import sys
        print(f"Step {i+1}/{self.args.num_steps}", file=sys.stderr, flush=True)
        # Render
        if self.args.show_train_run:
          self.show_image(curr_raw_image, curr_state_image)

        # Pick an action
        if random.random() < self.epsilon:
          action = torch.tensor([random.randint(0, 3)])
        else:
          action = self.dqn.predict_action(curr_state_image)

        # Step the environment
        _, done, reward, _ = self.arena.step(action[0])

        # Get the next state
        if done:
          next_raw_image = None
          next_state_image = None
        else:
          next_raw_image = self.arena.render()
          next_state_image = self.arena.render_torch_tensor(image=next_raw_image)

        # Record the transition in memory buffer
        transition = Transition(curr_state_image, action, next_state_image, reward)
        self.dqn.observe_transition(transition)

        # Update the model
        loss = self.dqn.optimize_model()
        sum_loss += loss
        optimize_count += 1

        # Update the next state
        if done:
          if reward > 0: success += 1
          else: failure += 1
          break
        else:
          curr_raw_image = next_raw_image
          curr_state_image = next_state_image

      # Update the target net
      if episode_i % self.args.target_update == 0:
        self.dqn.update_target()

      # Print information
      success_rate = (success / (episode_i + 1)) * 100.0
      avg_loss = sum_loss / optimize_count
      iterator.set_description(f"[Train Epoch {epoch}] Avg Loss: {avg_loss}, Success: {success}/{episode_i + 1} ({success_rate:.2f}%)")
    wandb.log({
      "train_loss": avg_loss,
      "train_success_rate": success_rate
    })

  def test_epoch(self, epoch):
    success, failure = 0, 0
    iterator = tqdm(range(self.args.num_test_episodes))
    for episode_i in iterator:
      _ = self.arena.reset()
      raw_image = self.arena.render()
      state_image = self.arena.render_torch_tensor(image=raw_image)
      for i in range(self.args.num_steps):
        import sys
        print(f"Step {i+1}/{self.args.num_steps}", file=sys.stderr, flush=True)
        # Show image
        if args.show_test_run:
          self.show_image(raw_image, state_image)

        # Pick an action
        action = self.dqn.predict_action(state_image)
        _, done, reward, _ = self.arena.step(action[0])
        raw_image = self.arena.render()
        state_image = self.arena.render_torch_tensor(image=raw_image)

        # Update the next state
        if done:
          if reward > 0: success += 1
          else: failure += 1
          break

      # Print information
      success_rate = (success / (episode_i + 1)) * 100.0
      iterator.set_description(f"[Test Epoch {epoch}] Success {success}/{episode_i + 1} ({success_rate:.2f}%)")
    wandb.log({
      "val_success_rate": success_rate
    })

    # save Model
    #torch.save(self.dqn.policy_net.state_dict(), f"{FILE_DIR}/checkpoints/model.pt")

  def run(self):
    self.test_epoch(0)
    for i in range(1, self.num_epochs + 1):
      print("Epoch", i)
      self.train_epoch(i)
      self.test_epoch(i)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--grid-x", type=int, default=5)
  parser.add_argument("--grid-y", type=int, default=5)
  parser.add_argument("--cell-size", type=float, default=0.5)
  parser.add_argument("--dpi", type=int, default=80)
  parser.add_argument("--batch-size", type=int, default=24)
  parser.add_argument("--num-enemies", type=int, default=5)
  parser.add_argument("--num-epochs", type=int, default=1000)
  parser.add_argument("--num-train-episodes", type=int, default=50)
  parser.add_argument("--num-test-episodes", type=int, default=50)
  parser.add_argument("--num-steps", type=int, default=30)
  parser.add_argument("--target-update", type=int, default=10)
  parser.add_argument("--epsilon", type=float, default=0.9)
  parser.add_argument("--epsilon-falloff", type=float, default=0.98)
  parser.add_argument("--gamma", type=float, default=0.999)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--replay-memory-capacity", type=int, default=3000)
  parser.add_argument("--seed", type=int, default=1357)
  parser.add_argument("--cuda", action="store_true")
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--attenuation", type=float, default=0.95)
  parser.add_argument("--show-run", action="store_true")
  parser.add_argument("--show-train-run", action="store_true")
  parser.add_argument("--show-test-run", action="store_true")
  parser.add_argument("--show-run-interval", type=int, default=0.001)
  parser.add_argument("--window-size", type=int, default=200)
  parser.add_argument("--overlay-prediction", action="store_true")
  parser.add_argument("--easy", action="store_true")
  parser.add_argument("--checkpoint", type=str, default=None)
  parser.add_argument("--switchover", type=int, default=1000)
  parser.add_argument("--lobster", action='store_true')
  args = parser.parse_args()



  # Set parameters
  args.show_run_interval = max(0.001, args.show_run_interval) # Minimum 1ms
  if args.show_run:
    args.show_train_run = True
    args.show_test_run = True
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  wandb.init(
      project="pacman",
      config=vars(args))

  if args.lobster:
    import os
    os.environ["USE_ARENA"] = "1"
    os.environ["ARENA_SIZE"] = "7"
    os.environ["SUBSPACE"] = "1"
    os.environ["CUB"] = "1"

  # Train
  trainer = Trainer(args.grid_x, args.grid_y, args.cell_size, args.dpi, args.num_enemies, args.epsilon, args.switchover, args)
  trainer.run()
  state_dicts = (trainer.dqn.policy_net.state_dict(), trainer.dqn.target_net.state_dict())

  print("Switching over to tripled grid size")
  args.grid_x *= 3
  args.grid_y *= 3
  args.num_epochs -= args.switchover

  if args.lobster:
    import os
    os.environ["STRATUM"]="16,17"
    os.environ["REORDER"]="13<-3"
    os.environ["NO_CHECK"] = "1"
  trainer = Trainer(args.grid_x, args.grid_y, args.cell_size, args.dpi, args.num_enemies, args.epsilon, args.num_epochs, args);
  trainer.dqn.policy_net.load_state_dict(state_dicts[0])
  trainer.dqn.target_net.load_state_dict(state_dicts[1])
  trainer.run()
