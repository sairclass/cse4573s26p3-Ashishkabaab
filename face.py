'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


from pyexpat import model

import torch

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: torch.Tensor) -> List[List[float]]:
    """
    Args:
        img : input image is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    detection_results: List[List[float]] = []

    ##### YOUR IMPLEMENTATION STARTS HERE #####
    #print(img.dtype)
    #print(img.shape)
    img_np = img.permute(1, 2, 0).to(torch.uint8).numpy() # Convert tensor to numpy array for face_recognition. Dimensions are CxHxW, but face_recognition expects HxWxC. Also convert to uint8 for face_recognition.
    face_locations = face_recognition.face_locations(img_np, model = "hog") # Detect faces using face_recognition
    
    for (top, right, bottom, left) in face_locations:
        box_width = right - left
        box_height = bottom - top
        # left = x
        # top = y
        detection_results.append([float(left), float(top), float(box_width), float(box_height)]) # Append detected bounding box to results

    return detection_results



def cluster_faces(imgs: Dict[str, torch.Tensor], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is a torch.Tensor represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    """
    Torch info: All intermediate data structures should use torch data structures or objects. 
    Numpy and cv2 are not allowed, except for face recognition API where the API returns plain python Lists, convert them to torch.Tensor.
    
    """
    cluster_results: List[List[str]] = [[] for _ in range(K)] # Please make sure your output follows this data format.
        
    ##### YOUR IMPLEMENTATION STARTS HERE #####
    
    #Get face encodings for each image
    encodings = []
    filenames = []

    for img_name, img_tensor in imgs.items():
        img_np = img_tensor.permute(1, 2, 0).to(torch.uint8).numpy() # Convert tensor to numpy array for face_recognition. Dimensions are CxHxW, but face_recognition expects HxWxC. Also convert to uint8 for face_recognition.
        face_locations = face_recognition.face_locations(img_np, model = "hog") # Detect faces using face_recognition
        if len(face_locations) == 0:
            # Use full image for images with no faces detected
            face_locations = [(0, img_np.shape[1], img_np.shape[0], 0)]

        enc_list = face_recognition.face_encodings(img_np, face_locations)
        if len(enc_list) == 0:
            # If no encodings are found, use a zero vector
            encoding = torch.zeros(128)
        else:
            encoding = torch.tensor(enc_list[0]) # Use the first face encoding for clustering
        filenames.append(img_name)
        encodings.append(encoding)
    
    N = len(encodings)
    encodings_tensor = torch.stack(encodings) # Stack encodings into a tensor of shape (N, 128)
    if N == 0:
        return cluster_results # Return empty clusters
    
    #Next: K-means clustering on the encodings

    # Initialize cluster centers randomly from the encodings
    indices = torch.randperm(N)[:K]
    centers = encodings_tensor[indices] # Shape (K, 128)

    iterations = 100 # Number of iterations for K-means
    prev_clusters = torch.full((N,), -1, dtype=torch.long) # Initialize previous cluster assignments to -1
    for _ in range(iterations):
        # Compute distances from encodings to cluster centers
        distances = torch.cdist(encodings_tensor, centers) # Shape (N, K)
        closest_clusters = torch.argmin(distances, dim=1) # Shape (N,)
        if torch.all(closest_clusters == prev_clusters): # If cluster assignments do not change, we have converged
            break
        prev_clusters = closest_clusters.clone()
        # Update cluster centers
        new_centers = []
        for k in range(K):
            cluster_points = encodings_tensor[closest_clusters == k]
            if len(cluster_points) > 0:
                new_centers.append(cluster_points.mean(dim=0))
            else:
                new_centers.append(centers[k])   # keep old center if empty
        centers = torch.stack(new_centers) # Shape (K, 128)
    labels = closest_clusters

    #Now build cluster results based on labels

    for i, fileName in enumerate(filenames):
        cluster_idx = labels[i].item()
        cluster_results[cluster_idx].append(fileName)
    return cluster_results


'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# TODO: Your functions. (if needed)

def pairwise_squared_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise squared Euclidean distance between two sets of vectors.
    Args:
        x: Tensor of shape (N, D)
        y: Tensor of shape (M, D)
    Returns:
        distances: Tensor of shape (N, M) where distances[i][j] is the squared Euclidean distance between x[i] and y[j].
    """
    # Using broadcasting to compute pairwise distances
    x_squared = torch.sum(x**2, dim=1).unsqueeze(1)  # Shape (N, 1)
    y_squared = torch.sum(y**2, dim=1).unsqueeze(0)  # Shape (1, M)
    cross_term = torch.mm(x, y.t())  # Shape (N, M)
    distances = x_squared + y_squared - 2 * cross_term  # Shape (N, M)
    return distances.clamp(min=0)  # Ensure non-negative distances
