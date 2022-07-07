import numpy as np
import numbers
import random
# from Tools.Denoise import edn
from scipy import spatial

# def ynoise(events, size):
#     ynoise = edn.Ynoise(10000, 3, 2, size[-1], size[-2])
#     res = ynoise.run(events[:, 1], events[:, 2], events[:, 3], events[:, 0])
#     events = events[res]
#     return events

def vnoise(events):
    txy = np.c_[[events[:, 0] / 1e3, events[:, 1], events[:, 2]]].T
    tree = spatial.KDTree(txy)
    neighbors = tree.query_ball_point(txy, r=2, p=float('inf'))

    def leastSquare(neighbor_idx):
        num_neighbors = len(neighbor_idx)
        if num_neighbors < 3:
            return 0, 0, 0
        else:
            nearest_neighbors = txy[neighbor_idx]
        return tuple(np.linalg.lstsq(np.c_[nearest_neighbors[:,1:],np.ones(num_neighbors)],nearest_neighbors[:,0],rcond=None)[0])

    vlsq = np.vectorize(leastSquare)
    result = vlsq(neighbors)
    v = np.power(result[0], 2) + np.power(result[1], 2)
    events = events[(v > 1e-3)&(v != np.nan)]
    # stream = dataset.transforms(stream)
    # stream = datatransforms.denoise(stream)
    return events

class RotatePointCloud(object):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    def __init__(self):
        pass

    def __call__(self, stream):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        stream[..., :3] = np.dot(stream[..., :3], rotation_matrix)
        return stream

class JitterPointCloud(object):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, stream):
        jittered_data = np.clip(self.sigma * np.random.randn(*stream.shape), -1*self.clip, self.clip)
        stream[:, :3] = jittered_data[:, :3] + stream[:, :3]
        return stream

class RandomCrop(object):
    """Crop the given stram sequences (n, 4) at a random location.
    Input:
        stream: event steam, size (n, 4). 
                4 : t, x, y, p
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif len(size) == 2:
            self.size = size
        elif len(size) > 2:
            self.size = (size[-2], size[-1])

    def __call__(self, stream):
        max_x = np.max(stream[:, 1])
        max_y = np.max(stream[:, 2])
        i = random.randint(0, max_x - self.size[0] + 1) if max_x - self.size[0] >= 0 else 0
        j = random.randint(0, max_y - self.size[1] + 1) if max_y - self.size[1] >= 0 else 0
        
        stream = stream[(stream[:, 1] >= i) & (stream[:, 1] < (self.size[0] + i))]
        stream = stream[(stream[:, 2] >= j) & (stream[:, 2] < (self.size[1] + j))]
        stream[:, (1, 2)] -= (i, j)
        return stream

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class CenterCrop(object):
    """Crop the given stram sequences (n, 4) at center location.
    Input:
        stream: event steam, size (n, 4). 
                4 : t, x, y, p
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif len(size) == 2:
            self.size = size
        elif len(size) > 2:
            self.size = (size[-2], size[-1])

    def __call__(self, stream):
        max_x, min_x = np.max(stream[:, 1]), np.min(stream[:, 1])
        max_y, min_y = np.max(stream[:, 2]), np.min(stream[:, 2])
        
        center_x = (max_x + min_x) // 2
        center_y = (max_y + min_y) // 2
        
        x_boundary = (center_x - self.size[0] // 2, center_x + self.size[0] // 2)
        y_boundary = (center_y - self.size[1] // 2, center_y + self.size[1] // 2)

        stream = stream[(stream[:, 1] >= x_boundary[0]) & (stream[:, 1] < x_boundary[1])]
        stream = stream[(stream[:, 2] >= y_boundary[0]) & (stream[:, 2] < y_boundary[1])]
        stream[:, (1, 2)] -= (x_boundary[0], y_boundary[0])
        return stream

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class RandomNoise(object):
    def __init__(self, p=0.5, size=(260, 346)):
        self.p = p
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        elif len(size) == 2:
            self.size = size
        elif len(size) > 2:
            self.size = (size[-2], size[-1])

    def __call__(self, stream):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        num = int(len(stream) * random.random() * self.p)
        t = np.random.rand(num) * np.max(stream[:, 0]).astype(np.float64)
        x = np.random.randint(0, self.size[0], size=num).astype(np.float64)
        y = np.random.randint(0, self.size[1], size=num).astype(np.float64)
        p = np.random.randint(2, size=num).astype(np.float16)
        noise = np.vstack((t, x, y, p)).transpose(1, 0)
        return np.concatenate((stream, noise))

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomLength(object):
    def __init__(self, p=0.8):
        self.p = p

    def __call__(self, stream):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        num = int(len(stream) * (random.random() * self.p + 1 - self.p))

        return stream[:num]

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomShift(object):
    def __init__(self, max_shift=20, resolution=(128, 128), ):
        self.max_shift = max_shift
        self.H, self.W = resolution

    def __call__(self, events, bounding_box=None):
        
        if bounding_box is not None:
            x_shift = np.random.randint(-min(bounding_box[0, 0], self.max_shift),
                                        min(self.W - bounding_box[2, 0], self.max_shift), size=(1,))
            y_shift = np.random.randint(-min(bounding_box[0, 1], self.max_shift),
                                        min(self.H - bounding_box[2, 1], self.max_shift), size=(1,))
            bounding_box[:, 0] += x_shift
            bounding_box[:, 1] += y_shift
        else:
            x_shift, y_shift = np.random.randint(-self.max_shift, self.max_shift+1, size=(2,))

        events[:, 1] += x_shift
        # events[:, 2] += y_shift

        valid_events = (events[:, 1] >= 0) & (events[:, 1] < self.W) & (events[:, 2] >= 0) & (events[:, 2] < self.H)
        events = events[valid_events]

        if bounding_box is None:
            return events

        return events, bounding_box


class RandomPerShift(object):
    def __init__(self, max_shift=20, resolution=(128, 128), ):
        self.max_shift = max_shift
        self.H, self.W = resolution

    def __call__(self, events, bounding_box=None):
        
        if bounding_box is not None:
            x_shift = np.random.randint(-min(bounding_box[0, 0], self.max_shift),
                                        min(self.W - bounding_box[2, 0], self.max_shift), size=(1,))
            y_shift = np.random.randint(-min(bounding_box[0, 1], self.max_shift),
                                        min(self.H - bounding_box[2, 1], self.max_shift), size=(1,))
            bounding_box[:, 0] += x_shift
            bounding_box[:, 1] += y_shift
        else:
            # x_shift, y_shift = np.random.randint(-self.max_shift, self.max_shift+1, size=(2,))
            # x_shift = np.random.randint(-self.max_shift, self.max_shift+1, size=(events.shape[0]))
            x_shift = np.random.randint(-self.max_shift, 1, size=(events.shape[0]))
            # y_shift = np.random.randint(-self.max_shift, self.max_shift+1, size=(events.shape[0]))

        events[:, 1] += x_shift
        # events[:, 2] += y_shift

        valid_events = (events[:, 1] >= 0) & (events[:, 1] < self.W) & (events[:, 2] >= 0) & (events[:, 2] < self.H)
        events = events[valid_events]

        if bounding_box is None:
            return events

        return events, bounding_box

class RandomHorizontalFlip(object):
    """Horizontally flip the given stream randomly with a given probability.
    Args:
        p (float): probability of the stream being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, stream):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        x_max = np.max(stream[:, 1])
        x_min = np.min(stream[:, 1])
        if random.random() < self.p:
            stream[:, 1] = x_max - stream[:, 1] + x_min
        return stream

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)