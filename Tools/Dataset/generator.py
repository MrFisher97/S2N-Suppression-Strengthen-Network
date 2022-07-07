import numpy as np
from scipy.optimize import leastsq

def leastsq_denoise(events, max_v=100, min_v=1e-4):
    if (events.shape[0] < 4):
        return False
    t, x, y = events[:, 0], events[:, 1], events[:, 2]

    def func(p, x0, x1):
        return p[0] * x0 + p[1] * x1 + p[2]

    def residuals(p, y, x0, x1):
        return y - func(p, x0, x1)

    p0 = [1, 1, 1]
    inv_v = leastsq(residuals, p0, args=(t, x, y))[0]
    v = 1 / inv_v[0]**2 + 1 / inv_v[1]**2
    if((v > max_v) or (v < min_v)):
        return False
    return True


class Generator():
    @staticmethod
    def generate_img(events, img_size, weight=None):
        '''
        two channels of cnts for positive and negtive events
        events: n * 4  
                4 : t, x, y, p
        img_size: H * W * C
        '''
        H, W, C = img_size
        t, x, y, p = events.T
        x = x.astype(np.int)
        y = y.astype(np.int)

        weight = 1 if weight is None else weight
        img = np.zeros((H * W, C), dtype="float32")
        if C == 2:
            np.add.at(img[:, 0], x[p == 0] + W * y[p == 0], weight)
            np.add.at(img[:, 1], x[p == 1] + W * y[p == 1], weight)
        elif C == 1:
            np.add.at(img, x + W * y, weight)

        img = img.reshape((H, W, C))
        return img

    def generate_voxel(events, ord='txyp', vox_size=(2, 16, 128, 128)):
        '''
        Generaete clip from events.
        parameter:
            events: ndarray of shape [num, channel].
            ord: ordering of the events tuple inside of events.
            vox_size: the size of generated clip [c, t, x, y].
        Returns:
            (CxTxWxH)(vox_size), numpy array of cnt/time frames with channels c
        '''
        t, x, y, p = np.split(events[:, (ord.find("t"), ord.find("x"), ord.find("y"), ord.find("p"))], 4, axis=1)
        C, T, H, W =  vox_size

        split_weight = t * 0.99 * T
        x = x.astype(np.uint32)
        y = y.astype(np.uint32)
        p = p.astype(bool)
        vox = np.zeros((C, T, H * W), dtype=np.float32)

        if C == 1:
            for i in range(T):
                weight = np.abs(split_weight - i)
                weight[weight > 1] = 0
                np.add.at(vox[0, i], x[p] + W * y[p], weight[p])
                np.add.at(vox[0, i], x[~p] + W * y[~p], -weight[~p])
        elif C == 2:
            for i in range(T):
                weight = np.abs(split_weight - i)
                weight[weight > 1] = 0
                np.add.at(vox[0, i], x[p] + W * y[p], weight[p])
                np.add.at(vox[1, i], x[~p] + W * y[~p], weight[~p])

        vox = vox.reshape((C, T, H, W))
        # clip = Filter.clip_hot_pixel_remove(clip.transpose(0, 3, 1, 2))
        # clip = clip.transpose(0, 2, 3, 1)
        vox = np.divide(vox, np.amax(vox, axis=(2, 3), keepdims=True),
                                out=np.zeros_like(vox),
                                where=vox!=0)
        return vox

    def generate_clip(events, ord='txyp', clip_size=(2, 16, 128, 128), split_by='cnt'):
        '''
        Generaete clip from events.
        parameter:
            events: ndarray of shape [num, channel].
            ord: ordering of the events tuple inside of events.
            clip_size: the size of generated clip [c, t, x, y].
                channel of output clip: 2:only cnt add; 
                                        3:contain average time (no polarity);
                                        4:contain average time (polarity)
            split_by: 'time' or 'cnt' decide how to split the events into even bins.
        Returns:
            (CxNxWxH)(clip_size), numpy array of cnt/time frames with channels
        '''

        t, x, y, p = np.split(events[:, (ord.find("t"), ord.find("x"), ord.find("y"), ord.find("p"))], 4, axis=1)
        C, T, H, W =  clip_size

        if split_by == 'time':
            split_weight = t * 0.99 * T
        elif split_by == 'cnt':
            split_weight = np.arange(0, 1, 1/events.shape[0]) * T
        
        x = x.astype(np.uint32)
        y = y.astype(np.uint32)
        p = p.astype(bool)
        split_index = split_weight.astype(np.uint32)

        clip = np.zeros((C, T * H * W), dtype=np.float32)
        np.add.at(clip[0], x[p] + W * y[p] + H * W * split_index[p], 1)
        np.add.at(clip[1], x[~p] + W * y[~p] + H * W * split_index[~p], 1)

        if C == 3:
            weight = split_weight - split_index
            weight[~p] *= -1
            np.add.at(clip[2], x + W * y + H  * W * split_index, weight)
            clip[2] = np.divide(clip[2], clip[0] + clip[1], out = np.zeros_like(clip[2]), where=clip[2]!=0)
        elif C == 4:
            weight = split_weight - split_index
            np.add.at(clip[2], x[p] + W * y[p] + H  * W * split_index[p], weight[p])
            np.add.at(clip[3], x[~p] + W * y[~p] + H  * W * split_index[~p], weight[~p])
            clip[2:] = np.divide(clip[2:], clip[:2], out = np.zeros_like(clip[2:]), where=clip[2:]!=0)

        clip = clip.reshape((C, T, H, W))
        # clip = Filter.clip_hot_pixel_remove(clip.transpose(0, 3, 1, 2))
        # clip = clip.transpose(0, 2, 3, 1)
        clip[:2] = np.divide(clip[:2], 
                                np.amax(clip[:2], axis=(2, 3), keepdims=True),
                                out=np.zeros_like(clip[:2]),
                                where=clip[:2]!=0)
        return clip