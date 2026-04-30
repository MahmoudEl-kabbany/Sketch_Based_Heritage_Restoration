import networkx as nx
import numpy as np

def _resolve_y_branches(graph):
    is_multi = hasattr(graph, "is_multigraph") and graph.is_multigraph()
    changed = True
    while changed:
        changed = False
        degrees = dict(graph.degree())
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
                
                if len(deg1_edges) == 2 and len(other_edges) == 1:
                    # Y-branch detected!
                    e1, e2 = deg1_edges[0], deg1_edges[1]
                    main_e = other_edges[0]
                    
                    n1 = e1[1] if e1[0] == node else e1[0]
                    n2 = e2[1] if e2[0] == node else e2[0]
                    main_node = main_e[1] if main_e[0] == node else main_e[0]
                    
                    p1 = np.array(graph.nodes[n1].get("o", [0.0, 0.0]))
                    p2 = np.array(graph.nodes[n2].get("o", [0.0, 0.0]))
                    midpoint = (p1 + p2) / 2.0
                    
                    # Concatenate points
                    main_pts = main_e[3].get("pts", []) if is_multi else main_e[2].get("pts", [])
                    main_pts = list(main_pts)
                    if len(main_pts) > 0:
                        # ensure it ends at node
                        if np.linalg.norm(np.array(main_pts[0]) - np.array(graph.nodes[node].get("o", [0,0]))) < np.linalg.norm(np.array(main_pts[-1]) - np.array(graph.nodes[node].get("o", [0,0]))):
                            main_pts = main_pts[::-1]
                    
                    new_pts = main_pts + [midpoint]
                    
                    # Remove nodes
                    graph.remove_node(n1)
                    graph.remove_node(n2)
                    graph.remove_node(node)
                    
                    # Create new node
                    new_node = max(graph.nodes()) + 1
                    graph.add_node(new_node, o=midpoint)
                    
                    # Add edge
                    if is_multi:
                        graph.add_edge(main_node, new_node, pts=np.array(new_pts))
                    else:
                        graph.add_edge(main_node, new_node, pts=np.array(new_pts))
                        
                    changed = True
                    break

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
_resolve_y_branches(G)
print("Nodes after:", G.nodes(data=True))
print("Edges after:", [(u, v, len(d.get('pts', []))) for u,v,k,d in G.edges(data=True, keys=True)])
