"""
Extracts and resizes the input screen
"""

from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

# default resized height
RESIZED_HEIGHT = 80

resize = T.Compose([T.ToPILImage(),
                    T.Resize(RESIZED_HEIGHT, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_screen(env):
    """Returns a resized screen of the game.
    For consistency and performance reasons, the size can be manually set by set_height.

    Args:
        env: the Gym environment.

    Returns:
        batch(1) * channel * height * width tensor: the screenshot tensor
    """
    # Returned screen requested by gym is 400x600x3
    # Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Convert to float, rescale, convert to torch tensor
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0)


def get_height():
    """Returns the current image height.
    """
    return resize.transforms[1].size


def set_height(height):
    """Sets the image height for resizing.

    Args:
        height (int): default is 80; original image height is 400.
    """
    resize.transforms[1].size = height


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import gym
    import torch
    env = gym.make('LunarLander-v2')
    env.reset()
    plt.figure()
    plt.imshow(get_screen(env).squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    plt.title('Example extracted screen')
    plt.show()
