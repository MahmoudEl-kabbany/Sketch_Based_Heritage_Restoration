import numpy as np
import cv2
from skimage.morphology import skeletonize
import sknw

def get_node_vectors(pts, node_id, is_start):
    # pts is sequence of coordinates.
    # if is_start, node is at pts[0]. vector is pts[5] - pts[0]
    idx = min(5, len(pts)-1)
    if is_start:
        vec = pts[idx] - pts[0]
    else:
        vec = pts[-idx-1] - pts[-1]
    norm = np.linalg.norm(vec)
    if norm < 1e-6: return np.array([0.0, 0.0])
    return vec / norm

def extract_macro_paths(graph):
    # build list of edges: (u, v, key, pts)
    edges = []
    for u, v, k, d in graph.edges(keys=True, data=True):
        pts = d.get('pts', [])
        if isinstance(pts, list): pts = np.array(pts, dtype=np.float64)
        else: pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 2 and pts.shape[1] >= 2:
            pts = pts[:, ::-1] # (y,x) to (x,y)
        edges.append((u, v, k, pts))

    # we want to pair edges at each node.
    node_edges = {}
    for i, (u, v, k, pts) in enumerate(edges):
        node_edges.setdefault(u, []).append((i, True)) # True if u is start of edge
        node_edges.setdefault(v, []).append((i, False)) # False if v is end of edge
        
    pairings = {} # edge_i -> (edge_j at start, edge_k at end)
    
    for node, incident in node_edges.items():
        if len(incident) == 2:
            # always merge degree 2
            e1, s1 = incident[0]
            e2, s2 = incident[1]
            pairings.setdefault(e1, {})[node] = (e2, s2)
            pairings.setdefault(e2, {})[node] = (e1, s1)
        elif len(incident) > 2:
            # find best pairs based on angle
            # sort pairs by dot product (closest to -1 is best)
            vecs = []
            for e, is_start in incident:
                pts = edges[e][3]
                vec = get_node_vectors(pts, node, is_start)
                vecs.append((e, is_start, vec))
                
            possible_pairs = []
            for i in range(len(vecs)):
                for j in range(i+1, len(vecs)):
                    dot = np.dot(vecs[i][2], vecs[j][2])
                    possible_pairs.append((dot, vecs[i], vecs[j]))
            
            # sort lowest dot product first (closest to 180 degrees)
            possible_pairs.sort(key=lambda x: x[0])
            used = set()
            for dot, v1, v2 in possible_pairs:
                e1, s1, _ = v1
                e2, s2, _ = v2
                # heuristic: if angle is too sharp (dot > 0.5), maybe don't merge unless it's a spur?
                # Actually let's just merge the best pair, leaving the rest.
                # If dot > 0.5 (less than 60 deg), they forms a very sharp V.
                # But to preserve uninterrupted line, we should merge.
                if e1 not in used and e2 not in used:
                    pairings.setdefault(e1, {})[node] = (e2, s2)
                    pairings.setdefault(e2, {})[node] = (e1, s1)
                    used.add(e1)
                    used.add(e2)

    # now extract paths
    visited_edges = set()
    macro_paths = []
    
    for i in range(len(edges)):
        if i in visited_edges: continue
        
        # trace back to start
        curr = i
        curr_dir = True # True means traversing edge from start to end
        # find the start of the macro path
        while True:
            node = edges[curr][0] if curr_dir else edges[curr][1]
            if node in pairings.get(curr, {}):
                next_e, next_s = pairings[curr][node]
                curr = next_e
                curr_dir = next_s # if it connects at its start, we will traverse it from end to start (False)
                if curr == i:
                    break # loop
            else:
                break
        
        # now trace forward
        start_curr = curr
        start_dir = curr_dir
        
        macro_pts = []
        is_loop = False
        
        while True:
            visited_edges.add(curr)
            pts = edges[curr][3]
            if not curr_dir: pts = pts[::-1]
            
            if len(macro_pts) > 0:
                # skip first point to avoid duplicate
                macro_pts.append(pts[1:])
            else:
                macro_pts.append(pts)
                
            node = edges[curr][1] if curr_dir else edges[curr][0]
            if node in pairings.get(curr, {}):
                next_e, next_s = pairings[curr][node]
                if next_e == start_curr:
                    is_loop = True
                    break
                
                curr = next_e
                curr_dir = not next_s # if it connects at start, we go start->end (True)
            else:
                break
                
        macro_pts = np.vstack(macro_pts)
        macro_paths.append((macro_pts, is_loop))
        
    return macro_paths


img = cv2.imread('test_images/restoration_test.png', cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
skeleton = skeletonize(binary / 255.0).astype(np.uint8)
graph = sknw.build_sknw(skeleton)
print("Graph edges:", len(graph.edges()))
paths = extract_macro_paths(graph)
print("Macro paths:", len(paths))
