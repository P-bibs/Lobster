import torch
from torch import nn
import itertools
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import scallopy

class PreNorm(nn.Module):
  def __init__(self, dim, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.fn = fn

  def forward(self, x, **kwargs):
    return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout = 0.):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, dim),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)


class Attention(nn.Module):
  def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
    super().__init__()
    inner_dim = dim_head *  heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.attend = nn.Softmax(dim = -1)
    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

    self.to_out = nn.Sequential(
      nn.Linear(inner_dim, dim),
      nn.Dropout(dropout)
    ) if project_out else nn.Identity()

  def forward(self, x):
    qkv = self.to_qkv(x).chunk(3, dim = -1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    attn = self.attend(dots)
    out = torch.matmul(attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)


class Transformer(nn.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
    super().__init__()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
        PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
        PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
      ]))

  def forward(self, x):
    for attn, ff in self.layers:
      x = attn(x) + x
      x = ff(x) + x
    return x


class ViT(nn.Module):
  def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
    super().__init__()
    image_height, image_width = pair(image_size)
    patch_height, patch_width = pair(patch_size)

    assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

    num_patches = (image_height // patch_height) * (image_width // patch_width)
    patch_dim = channels * patch_height * patch_width

    self.to_patch_embedding = nn.Sequential(
      Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
      nn.Linear(patch_dim, dim),
    )

    self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
    self.dropout = nn.Dropout(emb_dropout)

    self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
    self.to_latent = nn.Identity()

  def forward(self, img):
    x = self.to_patch_embedding(img)
    b, n, _ = x.shape

    x += self.pos_embedding[:, :n]
    x = self.dropout(x)

    x = self.transformer(x)

    x = self.to_latent(x)
    return x


def build_adj(num_block_x, num_block_y):
  adjacency = []
  block_coord_to_block_id = lambda x, y: y * num_block_x + x
  for i, j in itertools.product(range(num_block_x), range(num_block_y)):
    for (dx, dy) in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
      x, y = i + dx, j + dy
      if x >= 0 and x < num_block_x and y >= 0 and y < num_block_y:
        source_id = block_coord_to_block_id(i, j)
        target_id = block_coord_to_block_id(x, y)
        adjacency.append((source_id, target_id))
  return adjacency


class PathFinderNet(nn.Module):
  def __init__(self, provenance="difftopkproofs", k=3, num_block_x=6, num_block_y=6):
    super(PathFinderNet, self).__init__()

    # block
    self.num_block_x = num_block_x
    self.num_block_y = num_block_y
    self.num_blocks = num_block_x * num_block_y
    self.block_coord_to_block_id = lambda x, y: y * num_block_x + x

    # Adjacency
    self.adjacency = build_adj(num_block_x, num_block_y)

    # Scallop Context
    self.ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    self.ctx.add_relation("is_connected", ("i8", "i8"), input_mapping=self.adjacency)
    self.ctx.add_relation("is_endpoint", "i8", input_mapping=list(range(self.num_blocks)))
    self.ctx.add_rule("connected(x, y) = is_connected(x, y) \/ connected(x, z) /\ is_connected(z, y)")
    self.ctx.add_rule("endpoints_connected() = is_endpoint(x), is_endpoint(y), connected(x, y), x != y")
    self.connected = self.ctx.forward_function("endpoints_connected", output_mapping=())


class CNNPathFinder32Net(PathFinderNet):
  def __init__(self, provenance="difftopkproofs", k=3, num_block_x=6, num_block_y=6):
    super(CNNPathFinder32Net, self).__init__(provenance, k, num_block_x, num_block_y)

    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=5),
      nn.Conv2d(32, 32, kernel_size=5),
      nn.MaxPool2d(2),
      nn.Conv2d(32, 64, kernel_size=5),
      nn.Conv2d(64, 64, kernel_size=5),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )

    # Fully connected for `is_endpoint`
    self.is_endpoint_fc = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, self.num_blocks),
      nn.Sigmoid(),
    )

    # Fully connected for `connectivity`
    self.is_connected_fc = nn.Sequential(
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, len(self.adjacency)),
      nn.Sigmoid(),
    )

  def forward(self, image):
    embedding = self.cnn(image)
    is_connected = self.is_connected_fc(embedding)
    is_endpoint = self.is_endpoint_fc(embedding)
    result = self.connected(is_connected=is_connected, is_endpoint=is_endpoint)
    return result


class ViTPathFinderNet(PathFinderNet):
  def __init__(self, provenance="difftopkproofs", k=3, image_size=32, num_block_x=6, num_block_y=6, dim=256):
    super(ViTPathFinderNet, self).__init__(provenance, k, num_block_x, num_block_y)

    self.pad_x = (num_block_x - image_size % num_block_x) % num_block_x
    self.pad_y = (num_block_y - image_size % num_block_y) % num_block_y
    height, width = image_size + self.pad_x, image_size + self.pad_y
    self.encoder = ViT(image_size=(height, width), patch_size=(height // num_block_x, width // num_block_y), dim=dim, depth=4, heads=8, mlp_dim=512, channels=1, dim_head=32)
    self.is_endpoint_fc = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1), nn.Sigmoid())
    self.is_connected_fc = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim * 2), nn.GELU(), nn.Linear(dim * 2, 1), nn.Sigmoid())

  def forward(self, image):
    if self.pad_x or self.pad_y:
      pad_spec = (self.pad_y // 2, self.pad_y - self.pad_y // 2,
                  self.pad_x // 2, self.pad_x - self.pad_x // 2)
      image = torch.nn.functional.pad(image, pad_spec)
    embedding = self.encoder(image)
    src_idx, dst_idx = zip(*(self.adjacency))
    src_embed = embedding[:, src_idx]
    dst_embed = embedding[:, dst_idx]
    concat_embed = torch.cat((src_embed, dst_embed), dim=-1)
    is_connected = torch.squeeze(self.is_connected_fc(concat_embed), dim=-1)
    is_endpoint = torch.squeeze(self.is_endpoint_fc(embedding), dim=-1)
    result = self.connected(is_connected=is_connected, is_endpoint=is_endpoint)
    return result


def pair(t):
  return t if isinstance(t, tuple) else (t, t)
