import numpy as np
import cv2
import math
from scipy.ndimage import uniform_filter, convolve

def non_maximum_suppression(gradient_magnitude: np.ndarray, gradient_direction: np.ndarray) -> np.ndarray:
    pad_grad = np.pad(gradient_magnitude, pad_width=1)
    H = gradient_direction.shape[0]
    W = gradient_direction.shape[1]
    result = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            neighbour1 = 0
            neighbour2 = 0
            value = gradient_magnitude[i,j]
            ang = gradient_direction[i,j]
            if (22.5 <= ang <67.5):
                neighbour1 = pad_grad[i+2,j+2]
                neighbour2 = pad_grad[i,j]
            elif (67.5 <= ang <112.5):
                neighbour1 = pad_grad[i+2,j+1]
                neighbour2 = pad_grad[i,j+1]
            elif (112.5 <= ang <157.5):
                neighbour1 = pad_grad[i+2,j]
                neighbour2 = pad_grad[i,j+2]
            else:
                neighbour1 = pad_grad[i+1,j+2]
                neighbour2 = pad_grad[i+1,j]
            if (value>= neighbour1 and value>= neighbour2):
                result[i,j] = value
                
    return result


def image_to_good_soble_with_direction(input: np.ndarray) -> tuple [np.ndarray, np.ndarray]:
    grad_x = cv2.Sobel(input, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(input, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = (np.arctan2(grad_y, grad_x) * (180 / np.pi)) % 180
    return non_maximum_suppression(gradient_magnitude, gradient_direction), gradient_direction

def image_to_good_soble_per_colour_with_direction(input: np.ndarray) -> tuple [np.ndarray, np.ndarray]:
    channels = cv2.split(input)
    grad_x_ch = [cv2.Sobel(ch, cv2.CV_64F, 1, 0, ksize=3) for ch in channels]
    grad_y_ch = [cv2.Sobel(ch, cv2.CV_64F, 0, 1, ksize=3) for ch in channels]
    # combine across channels by Euclidean norm
    grad_x = np.sqrt(sum(gx**2 for gx in grad_x_ch))
    grad_y = np.sqrt(sum(gy**2 for gy in grad_y_ch))

    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = (np.arctan2(grad_y, grad_x) * (180 / np.pi)) % 180

    return non_maximum_suppression(gradient_magnitude, gradient_direction), gradient_direction


def proportionally_expand_edges(input: np.ndarray, directions: np.array,threshhold:float, scale:float) -> np.array:
    H,W = input.shape
    output = np.zeros((H,W))
    for i in range(round (H)):
        for j in range(round(W)):
            if input[i,j] > threshhold:
                output[i,j] = 255
                mag = input[i,j]
                xslope = math.sin(math.radians( directions[i,j]))
                yslope = math.cos(math.radians( directions[i,j]))
                for k in range(0, round(mag*scale)):
                    xcord = round( xslope * k)
                    ycord = round( yslope * k)
                    if (0 <= xcord + i < H and 0 <= ycord + j < W ):
                        output[xcord+i,ycord+j ] = 255
                    xcord *= -1
                    ycord *= -1
                    if (0 <= xcord + i < H and 0 <= ycord + j < W ):
                        output[xcord+i,ycord+j ] = 255

    return output



def image_to_good_soble(input: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(input, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(input, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_direction = (np.arctan2(grad_y, grad_x) * (180 / np.pi)) % 180
    return non_maximum_suppression(gradient_magnitude, gradient_direction)





def double_thresholding(image: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """
    Applies double thresholding to an edge image.
    For each pixel, if the gradient magnitude is higher than the high threshold,
    it is marked as a strong edge (255). If the gradient magnitude is between the low
    and high thresholds, it is marked as a weak edge (128). Otherwise, it is suppressed (0).

    Parameters:
        image (numpy.ndarray (H, W)): The gradient magnitude after NMS.
        low_threshold (float): Lower threshold for edge linking.
        high_threshold (float): Upper threshold for strong edges.

    Returns:
        numpy.ndarray (H, W): edge map
    """
    H = image.shape[0]
    W = image.shape[1]
    result = np.zeros((H,W))
    for i in range(H):
        for j in range(W):
            value = image[i,j]
            if (value >= high_threshold):
                result[i,j] = 255
            elif (value >= low_threshold):
                result[i,j] = 127
    return result



def splotch_map(edge_map: np.ndarray) -> np.ndarray:
    H, W = edge_map.shape
    result = np.zeros((H, W), dtype=int)
    
    class UF:
        def __init__(self):
            self.parent = {}
            self.size = {}

        def find(self, x):
            self.parent.setdefault(x, x)
            self.size.setdefault(x, 1)

            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])

            return self.parent[x]

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)

            if rx == ry:
                return

            # union by size
            if self.size[rx] < self.size[ry]:
                rx, ry = ry, rx

            self.parent[ry] = rx
            self.size[rx] += self.size[ry]
    
    uf = UF()
    label = 1

    for i in range(H):
        for j in range(W):
            if edge_map[i, j] != 0:
                continue
            neighbors = []
            if i > 0 and edge_map[i-1, j] == 0:
                neighbors.append(result[i-1, j])
            if j > 0 and edge_map[i, j-1] == 0:
                neighbors.append(result[i, j-1])
            if not neighbors:
                result[i, j] = label
                label += 1
            else:
                min_label = min(neighbors)
                result[i, j] = min_label
                for n in neighbors:
                    uf.union(n, min_label)
    
    # Flatten all equivalences
    for i in range(H):
        for j in range(W):
            if result[i, j] != 0:
                result[i, j] = uf.find(result[i, j])
    
    return result



def splat_map(edge_map: np.ndarray) -> np.ndarray:
    H, W = edge_map.shape
    result = np.zeros((H, W), dtype=int)
    
    class UF:
        def __init__(self):
            self.parent = {}
            self.size = {}

        def find(self, x):
            self.parent.setdefault(x, x)
            self.size.setdefault(x, 1)

            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])

            return self.parent[x]

        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)

            if rx == ry:
                return

            # union by size
            if self.size[rx] < self.size[ry]:
                rx, ry = ry, rx

            self.parent[ry] = rx
            self.size[rx] += self.size[ry]
    
    uf = UF()
    label = 1

    for i in range(H):
        for j in range(W):
            if edge_map[i, j] == 0:
                continue
            neighbors = []
            if i > 0 and edge_map[i-1, j] != 0:
                neighbors.append(result[i-1, j])
            if j > 0 and edge_map[i, j-1] != 0:
                neighbors.append(result[i, j-1])

            if j > 0 and i > 0 and edge_map[i-1, j-1] != 0:
                neighbors.append(result[i-1, j-1])
            
            if j < (W-1) and i > 0 and edge_map[i-1, j+1] != 0:
                neighbors.append(result[i-1, j+1])

            if not neighbors:
                result[i, j] = label
                label += 1
            else:
                min_label = min(neighbors)
                result[i, j] = min_label
                for n in neighbors:
                    uf.union(n, min_label)
    
    # Flatten all equivalences
    for i in range(H):
        for j in range(W):
            if result[i, j] != 0:
                result[i, j] = uf.find(result[i, j])
    
    return result



def fill_in_the_blank(edge_map: np.ndarray):
    # Kernel for counting 8-neighbours
    kernel = np.ones((3,3), dtype=np.uint8)

    # Count neighbours (including self, so subtract later)
    neighbours = cv2.filter2D((edge_map > 0).astype(np.uint8), -1, kernel)
    neighbours = neighbours - (edge_map > 0).astype(np.uint8)

    # Create output
    output = edge_map.copy()
    output[(edge_map == 0) & (neighbours > 2)] = 255

    return output 

def extend_lines_piercing(edge_map: np.ndarray,direction_array: np.ndarray, length:int):
    H,W = edge_map.shape
    new_edge_map = edge_map.copy()
    new_direction_array = direction_array.copy()
    for i in range(H):
        for j in range(W):
            if (edge_map[i,j] > 0 ):
                ang = direction_array[i,j]
                pi_ang = ang *np.pi /180
                val = edge_map[i,j]
                _cos = math.cos(pi_ang)
                _sin = math.sin(pi_ang)
                scal_length = round( length* val/255)

                filling = False

                for k in range( scal_length):
                    _dis = (scal_length- k)
                    ti = round( i+ (_dis+1)*_cos )
                    tj = round(j+ (_dis+1)*_sin )

                    if (ti >= 0 and ti < H and tj >= 0 and tj < W):
                        if (edge_map[ti,tj] > 0):
                            filling = True
                        else:
                            if filling:
                                new_edge_map[ti,tj] = val
                                new_direction_array[ti,tj] = ang

                filling = False
                for k in range(scal_length):
                    _dis = (scal_length- k)
                    ti = round(i -(_dis+1)*_cos )
                    tj = round(j -(_dis+1)*_sin )
                    
                    if (ti >= 0 and ti < H and tj >= 0 and tj < W):
                        if (edge_map[ti,tj] > 0):
                            filling = True
                        else:
                            if filling:
                                new_edge_map[ti,tj] = val
                                new_direction_array[ti,tj] = ang
    return new_edge_map,new_direction_array

def endpoints(edge_map: np.ndarray):
    kernel = np.ones((3,3), dtype=np.uint8)
    kernel[1,1] = 0
    neighbor_count = convolve(
        (edge_map > 0).astype(np.uint8),
        kernel,
        mode="constant"
    )
    arr_end = (edge_map > 0) & (neighbor_count <= 1)
    return edge_map * arr_end


def extend_lines(edge_map: np.ndarray,direction_array: np.ndarray, length:int,angle_pierce:float):
    H,W = edge_map.shape
    new_edge_map = edge_map.copy()
    new_direction_array = direction_array.copy()
    for i in range(H):
        for j in range(W):
            if (edge_map[i,j] > 0 ):
                ang = direction_array[i,j]
                pi_ang = ang *np.pi /180
                _cos = math.cos(pi_ang)
                _sin = math.sin(pi_ang)
                val = edge_map[i,j]
                end = -1
                scal_length =  length
                for k in range( scal_length):
                    ti = int(round(i + (k+1)*_cos))
                    tj = int(round(j + (k+1)*_sin))

                    if (ti >= 0 and ti < H and tj >= 0 and tj < W):
                        if (edge_map[ti,tj] > 0):

                            diff = abs(ang - direction_array[ti,tj]) % 180 
                            if diff > 90:
                                diff = 180 - diff
                            
                            if diff <= angle_pierce:
                                end = k
                            if diff > angle_pierce:
                                break
                if (end!= -1):
                    for k in range(end):
                        ti = int( i+ (k+1)*_cos)
                        tj = int(j+ (k+1)*_sin )
                        new_edge_map[ti,tj] = val
                        new_direction_array[ti,tj] = ang

                end = -1
                for k in range( scal_length):
                    ti = int( i- (k+1)*_cos )
                    tj = int(j- (k+1)*_sin )

                    if (ti >= 0 and ti < H and tj >= 0 and tj < W):
                        if (edge_map[ti,tj] > 0):
                            end = k
                            diff = abs(ang - direction_array[ti,tj]) % 180 
                            if diff > 90:
                                diff = 180 - diff
                            if diff > angle_pierce:
                                break
                if (end!= -1):
                    for k in range(end):
                        ti = int( i-(k+1)*_cos)
                        tj = int(j  -(k+1)*_sin )
                        new_edge_map[ti,tj] = val
                        new_direction_array[ti,tj] = ang
    return new_edge_map,new_direction_array
    
                     




def connect_lines(edge_map: np.ndarray, kernel_size: int = 3):
    # binarize: 0 or 255
    binary = (edge_map > 0).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return connected





def adaptive_double_threshold(grad_mag, window_size=15, low_frac=0.9, high_frac=1.5, min_thresh=10,max_thresh=20):
    """
    Adaptive double threshold based on local edge density.
    
    Parameters:
        grad_mag (np.ndarray): 2D gradient magnitude map
        window_size (int): size of local window for averaging
        low_frac (float): fraction of local mean for low threshold
        high_frac (float): fraction of local mean for high threshold
        min_thresh (float): minimum threshold to avoid noise
        
    Returns:
        np.ndarray: single uint8 array with strong edges = 255, weak = 127, rest = 0
    """
    # Local mean of gradient magnitude (measures density of detail)
    local_mean = uniform_filter(grad_mag, size=window_size, mode='reflect')
    
    # Thresholds scale with local density
    low_thresh  = np.clip(low_frac * local_mean, min_thresh,max_thresh)
    high_thresh = np.clip(high_frac * local_mean, min_thresh,2*max_thresh)
    
    # Masks
    strong_edges = grad_mag >= high_thresh
    weak_edges   = (grad_mag >= low_thresh) & (grad_mag < high_thresh)
    
    # Combine into single output
    return (255*strong_edges + 127*weak_edges).astype(np.uint8)