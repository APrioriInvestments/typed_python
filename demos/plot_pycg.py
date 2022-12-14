import networkx as nx
import json
import matplotlib.pyplot as plt
from pyvis.network import Network

color_filter = {
    "menus": "red",
    "loader": "purple",
    "forms": "darkblue",
    "popups": "blue",
    "default": "black",
}


def to_ntwx_json(data: dict) -> nx.DiGraph:

    nt = nx.DiGraph()

    def _ensure_key(name):
        if name not in nt:
            nt.add_node(name, size=50)

    for node in data:
        _ensure_key(node)
        for child in data[node]:
            _ensure_key(child)
            nt.add_edge(node, child)
    return nt


def ntw_pyvis(ntx: nx.DiGraph, root, size0=5, loosen=2):
    nt = Network(width="1200px", height="800px", directed=True)
    for node in ntx.nodes:
        mass = ntx.nodes[node]["size"] / (loosen * size0)
        size = size0 * ntx.nodes[node]["size"] ** 0.5
        label = node
        color = color_filter["default"]
        for key in color_filter:
            if key in node:
                color = color_filter[key]
        kwargs = {
            "label": label,
            "mass": mass,
            "size": size,
            "color": color,
        }
        nt.add_node(
            node,
            **kwargs,
        )

    for link in ntx.edges:
        try:
            depth = nx.shortest_path_length(ntx, source=root, target=link[0])
            width = max(size0, size0 * (12 - 4 * depth))
        except:
            width = 5

        nt.add_edge(link[0], link[1], width=width)

    nt.show_buttons(filter_=["physics"])
    nt.show("nodes.html")


if __name__ == "__main__":
    JSON_PATH = "./cg.json"

    with open(JSON_PATH) as flines:
        cgdata = json.load(flines)

    ntx = to_ntwx_json(cgdata)
    # print(ntx.number_of_nodes())

    ntw_pyvis(ntx, root=0)
    # nx.draw(ntx)
    # plt.show()
