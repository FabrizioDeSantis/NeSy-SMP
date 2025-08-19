import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from pyvis.network import Network
from rdflib.namespace import split_uri
import matplotlib.pyplot as plt
from rdflib import URIRef
import networkx as nx

def visualize(graph):
    with open("kg.txt", "w") as f:
        for s, p, o in graph:
            s = s.replace("http://example.org/process/", "")
            p = p.replace("http://example.org/process/", "")
            o = o.replace("http://example.org/process/", "")
            
            s = s.replace("http://snomed.info/id/", "")
            p = p.replace("http://snomed.info/id/", "")
            o = o.replace("http://snomed.info/id/", "")
            
            # _, s = split_uri(s)
            # _, p = split_uri(p)
            # if isinstance(o, URIRef):
            #     _, o = split_uri(o)
            print(f"Subject: {s}, Predicate: {p}, Object: {o}")
            f.write(f"{s}\t{p}\t{o}\n")
    nx_graph = rdflib_to_networkx_multidigraph(graph)
    

    # Strip namespaces from node labels
    for node in nx_graph.nodes():
        if isinstance(node, URIRef):
            try:
                _, local_part = split_uri(node)
                nx_graph.nodes[node]["label"] = local_part
            except:
                nx_graph.nodes[node]["label"] = str(node)
        else:
            nx_graph.nodes[node]["label"] = str(node)
        nx_graph.nodes[node]["size"] = 40  # Increase node size here
        nx_graph.nodes[node]["font"] = {"size": 40} 
        # if nx_graph.nodes[node]["label"] == "Sepsis" or nx_graph.nodes[node]["label"] == "Patient" or nx_graph.nodes[node]["label"] == "Outcome":
        #     nx_graph.nodes[node]["color"] = "#ff0000"
        # elif nx_graph.nodes[node]["label"] == "HighLactateIncreasesMortality":
        #     nx_graph.nodes[node]["color"] = "#16cd00"
        # elif nx_graph.nodes[node]["label"] == "Symptom":
        #     nx_graph.nodes[node]["color"] = "#2edcff"
        # elif nx_graph.nodes[node]["label"] == "Disease":
        #     nx_graph.nodes[node]["color"] = "#ffb62e"

    # Strip namespaces from edge labels
    for u, v, k, data in nx_graph.edges(data=True, keys=True):
        try:
            _, local_part = split_uri(k)
            label = local_part
        except:
            label = str(k)
        data["label"] = label
        if label == "is_followed_by":
            data["color"] = "#16cd00"  # Set edge color to red
        else:
            data["color"] = "#848484"  # Set edge color to gray
        print(data)

    net = Network(
        height="1000px",
        width="100%",
        notebook=False,
        directed=True
    )

    plt.figure(figsize=(100, 100))
    nx.draw(nx_graph, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=16)

    # Save to PNG
    plt.savefig("graph.svg", format="svg")
    plt.close()

    net.from_nx(nx_graph)

    net.barnes_hut(spring_length=400, spring_strength=0.05, damping=0.09)
    net.repulsion(node_distance=300, central_gravity=0.01, spring_length=500)

    net.show_buttons()
    net.write_html("clinical_knowledge_graph.html")