import networkx as nx
import numpy as np

def _prune_skeleton_spurs(graph, threshold_length=15.0):
    is_multi = hasattr(graph, "is_multigraph") and graph.is_multigraph()
    
    # 1. Merge Y-branches at endpoints
    changed = True
    while changed:
        changed = False
        degrees = dict(graph.degree())
        # Find a degree 3 node connected to two degree 1 nodes
        for node, deg in degrees.items():
            if deg == 3:
                adj_edges = list(graph.edges(node, keys=True, data=True) if is_multi else graph.edges(node, data=True))
                deg1_edges = []
                other_edges = []
                for ae in adj_edges:
                    other = ae[1] if ae[0] == node else ae[0]
                    if degrees.get(other, 0) == 1:
                        deg1_edges.append(ae)
                    else:
                        other_edges.append(ae)
                
                if len(deg1_edges) == 2:
                    # Y-branch detected!
                    e1, e2 = deg1_edges[0], deg1_edges[1]
                    n1 = e1[1] if e1[0] == node else e1[0]
                    n2 = e2[1] if e2[0] == node else e2[0]
                    
                    p1 = np.array(graph.nodes[n1].get("o", [0.0, 0.0]))
                    p2 = np.array(graph.nodes[n2].get("o", [0.0, 0.0]))
                    midpoint = (p1 + p2) / 2.0
                    
                    # Remove the two degree-1 nodes and edges
                    graph.remove_node(n1)
                    graph.remove_node(n2)
                    
                    # Create a new node at midpoint
                    new_node = max(graph.nodes()) + 1
                    graph.add_node(new_node, o=midpoint)
                    
                    # Add edge from junction to midpoint
                    # For simplicity, we just add a straight line pts
                    pts1 = e1[3].get("pts", []) if is_multi else e1[2].get("pts", [])
                    pts_new = np.array([graph.nodes[node].get("o", [0.0, 0.0]), midpoint])
                    if is_multi:
                        graph.add_edge(node, new_node, pts=pts_new)
                    else:
                        graph.add_edge(node, new_node, pts=pts_new)
                        
                    changed = True
                    break # Restart while loop because graph changed
                    
    # 2. Existing prune logic
    changed = True
    while changed:
        changed = False
        degrees = dict(graph.degree())
        edges_to_remove = []

        edges = graph.edges(keys=True, data=True) if is_multi else graph.edges(data=True)

        for edge in edges:
            u, v = edge[0], edge[1]
            deg_u, deg_v = degrees.get(u, 0), degrees.get(v, 0)

            if (deg_u == 1 and deg_v > 1) or (deg_v == 1 and deg_u > 1):
                junction_node = v if deg_u == 1 else u
                spur_node = u if deg_u == 1 else v
                junction_deg = degrees.get(junction_node, 0)
                
                data = edge[3] if is_multi else edge[2]
                pts = data.get("pts", [])

                if len(pts) > 1:
                    pts_arr = np.array(pts)
                    diffs = np.diff(pts_arr, axis=0)
                    length = np.sum(np.linalg.norm(diffs, axis=1))
                else:
                    length = 0.0
                
                effective_threshold = threshold_length
                if junction_deg == 3:
                    adj_edges = graph.edges(junction_node, keys=True, data=True) if is_multi else graph.edges(junction_node, data=True)
                    other_edges = []
                    for ae in adj_edges:
                        if ae[0] == spur_node or ae[1] == spur_node:
                            continue
                        other_edges.append(ae)
                    
                    if len(other_edges) == 2:
                        vecs = []
                        j_pt = np.array(graph.nodes[junction_node].get("o", [0.0, 0.0]), dtype=np.float64)
                        for ae in other_edges:
                            ae_data = ae[3] if is_multi else ae[2]
                            ae_pts = ae_data.get("pts", [])
                            if len(ae_pts) >= 2:
                                p_arr = np.array(ae_pts, dtype=np.float64)
                                dist_start = np.linalg.norm(p_arr[0] - j_pt)
                                dist_end = np.linalg.norm(p_arr[-1] - j_pt)
                                if dist_start < dist_end:
                                    k = min(5, len(p_arr) - 1)
                                    vecs.append(p_arr[k] - p_arr[0])
                                else:
                                    k = max(0, len(p_arr) - 1 - 5)
                                    vecs.append(p_arr[k] - p_arr[-1])
                        
                        if len(vecs) == 2:
                            n1, n2 = np.linalg.norm(vecs[0]), np.linalg.norm(vecs[1])
                            if n1 > 1e-6 and n2 > 1e-6:
                                cos_theta = np.clip(np.dot(vecs[0], vecs[1]) / (n1 * n2), -1.0, 1.0)
                                angle = np.degrees(np.arccos(cos_theta))
                                if angle > 130.0:
                                    effective_threshold = min(4.0, threshold_length * 0.35)
                                else:
                                    effective_threshold = threshold_length * 1.5
                elif junction_deg > 3:
                    effective_threshold = threshold_length * 0.75

                if length <= effective_threshold:
                    edges_to_remove.append(edge)

        for edge in edges_to_remove:
            if is_multi:
                graph.remove_edge(edge[0], edge[1], key=edge[2])
            else:
                graph.remove_edge(edge[0], edge[1])
            changed = True

# Test code
G = nx.MultiGraph()
G.add_node(0, o=np.array([0, 0]))
G.add_node(1, o=np.array([0, 10]))
G.add_node(2, o=np.array([-5, 15]))
G.add_node(3, o=np.array([5, 15]))

# Y-branch at node 1
G.add_edge(0, 1, pts=np.array([[0,0], [0,10]]))
G.add_edge(1, 2, pts=np.array([[0,10], [-5,15]]))
G.add_edge(1, 3, pts=np.array([[0,10], [5,15]]))

print("Nodes before:", G.nodes(data=True))
_prune_skeleton_spurs(G)
print("Nodes after:", G.nodes(data=True))
print("Edges after:", G.edges(data=True))
