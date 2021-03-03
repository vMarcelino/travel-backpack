from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
try:
    from typing import Literal
except:
    from typing_extensions import Literal

from travel_backpack.exceptions import check_and_raise
from travel_backpack.variables import ensure_type

T = TypeVar('T')
TG = TypeVar('TG', bound='GraphObject')
TN = TypeVar('TN', bound='Node')


class GraphObject:
    objects: List[TG] = []
    object_count = 0

    def __init__(self, object_base_class: Type[TG], label: Optional[str] = None):
        instance_class = type(self)
        self.deleted = False
        if label is None:
            label = cast(str, instance_class.__name__)

        self.label = label

        if object_base_class.objects == []:
            # initialize the object
            object_base_class.objects = []
            object_base_class.object_count = 0

        object_base_class.objects.append(self)

        self.graph_id = GraphObject.object_count
        GraphObject.object_count += 1

        self.class_id = object_base_class.object_count
        object_base_class.object_count += 1
        del_func = self.delete

        def start_delete(*args, **kwargs):
            if not self.deleted:
                return del_func(*args, **kwargs)

        # self.delete = start_delete
        setattr(self, 'delete', start_delete)  # in this form to avoid linting errors

    def __hash__(self):
        return hash(self.graph_id)

    def __str__(self):
        return f'{self.class_id}\n{self.label}'

    def __repr__(self):
        return f'{self.label}'

    @property
    def query(self):
        return Query(start=self)

    def delete(self):
        raise NotImplementedError

    @overload
    def ensure_type(self, t: Type[TG], output_type: None = None) -> TG:
        ...

    @overload
    def ensure_type(self, t: Type, output_type: Type[TG]) -> TG:
        ...

    def ensure_type(self, t: Type[TG], output_type: Optional[Type[TG]] = None) -> TG:
        return ensure_type(self, t, output_type)


class EdgeLabelNotAllowed(Exception):
    ...


TG_co = TypeVar('TG_co', bound=GraphObject, covariant=True)
LabelType = Union[str, Type[TG_co]]
LabelListType = Union[Iterable[LabelType], Callable[[], Iterable[LabelType]]]


class Edge(GraphObject):
    # whitelist or blacklist
    __list_mode__: Literal['whitelist', 'blacklist'] = 'whitelist'
    # if defined, all listed here will be in the list,
    # regardless if it's source or destination.
    # if None, disconsider this check
    __node_list__: Optional[LabelListType] = None
    __source_list__: Optional[LabelListType] = None
    __destination_list__: Optional[LabelListType] = None

    def __init__(self, label: str, source: Node, destination: Node):
        super().__init__(Edge, label=label)
        self._source = None
        self._destination = None

        self.source = source
        self.destination = destination

    @property
    def source(self) -> Node:
        return ensure_type(self._source, Node)

    @source.setter
    def source(self, new_source: Node):
        if self._is_label_allowed(new_source.label, self.__source_list__) and \
           new_source._is_edge_allowed(self.label, inbound=False):
            if self._source is not None:
                old_source = self.source
                old_source.outbound.remove(self)
            new_source.outbound.add(self)
            self._source = new_source
        else:
            raise EdgeLabelNotAllowed(f'Outbound edge label is not allowed on this node: {self.label}')

    @property
    def destination(self) -> 'Node':
        return ensure_type(self._destination, Node)

    @destination.setter
    def destination(self, new_destination: 'Node'):
        if self._is_label_allowed(new_destination.label, self.__destination_list__) and \
           new_destination._is_edge_allowed(self.label, inbound=True):
            if self._destination is not None:
                old_destination = self.destination
                old_destination.inbound.remove(self)
            new_destination.inbound.add(self)
            self._destination = new_destination
        else:
            raise EdgeLabelNotAllowed(f'Inbound edge label is not allowed on this node: {self.label}')

    def _is_label_allowed(self, label: str, label_list: Optional[LabelListType]) -> bool:
        _all_labels = label_list_to_str_list(self.__node_list__)
        if _all_labels is None:
            _all_labels = []

        _label_list = label_list_to_str_list(label_list)

        if _label_list is None:
            return True

        _label_list += _all_labels
        if self.__list_mode__ == 'whitelist':
            allowed_labels = _label_list
            return label in allowed_labels
        else:
            forbidden_labels = _label_list
            return not (label in forbidden_labels)

    def delete(self):
        self.source.outbound.remove(self)
        self.destination.inbound.remove(self)
        Edge.objects.remove(self)
        self.deleted = True
        # self.source = None
        # self.destination = None


def label_type_to_str(label: LabelType) -> str:
    if isinstance(label, str):
        return label
    elif issubclass(label, GraphObject):
        return label.label
    else:
        raise TypeError(f'Expected either a str or a GraphObject. Got, {type(label)} {label}')


def label_list_to_str_list(lst: Optional[LabelListType]) -> Optional[List[str]]:
    op_label_list: Optional[Iterable[LabelType]]
    if isinstance(lst, Callable):
        op_label_list = lst()  # type: ignore
    elif isinstance(lst, Iterable):
        op_label_list = lst
    elif lst is None:
        op_label_list = None
    else:
        raise TypeError(f'Label list is not of supported type. Is {type(lst)}')

    if op_label_list is not None:
        label_list: List[str] = []
        for element in op_label_list:
            label_list.append(label_type_to_str(element))
        return label_list
    else:
        return None


class Node(GraphObject):
    # whitelist or blacklist
    __list_mode__: Literal['whitelist', 'blacklist'] = 'whitelist'

    # if defined, all listed here will be in the list,
    # regardless if it's inbound or outbound.
    # if None, disconsider this check
    __edge_list__: Optional[LabelListType] = None

    __inbound_edge_list__: Optional[LabelListType] = None
    __outbound_edge_list__: Optional[LabelListType] = None

    def __init__(self, label: str = None):
        super().__init__(Node, label=label)

        self.inbound: Set[Edge] = set()
        self.outbound: Set[Edge] = set()

    def add_edge(self, edge_label: Union[str, Edge, Type[Edge]], to: Node, two_way=False):
        if isinstance(edge_label, Edge):
            edge_label = edge_label.label

        elif (not isinstance(edge_label, str)) and issubclass(edge_label, Edge):
            edge_label = edge_label.__name__

        # will raise an error if edge label is not allowed
        edge = Edge(source=self, destination=to, label=edge_label)

        if two_way:
            # we don't want to error and
            # leave the operation half-done
            try:
                to.add_edge(edge_label=edge_label, to=self, two_way=False)

            except EdgeLabelNotAllowed:
                edge.delete()
                raise

    def _is_edge_allowed(self, label: str, inbound: bool):
        is_whitelist = self.__list_mode__ == 'whitelist'

        _all = label_list_to_str_list(self.__edge_list__)
        _in = label_list_to_str_list(self.__inbound_edge_list__)
        _out = label_list_to_str_list(self.__outbound_edge_list__)

        if _all is not None:
            if inbound:
                if _in is not None:
                    return label in _all + _in if is_whitelist else label not in _all + _in
                else:
                    return label in _all if is_whitelist else label not in _all  # pylint: disable=unsupported-membership-test
            else:
                if _out is not None:
                    return label in _all + _out if is_whitelist else label not in _all + _out
                else:
                    return label in _all if is_whitelist else label not in _all  # pylint: disable=unsupported-membership-test
        else:
            if inbound:
                if _in is not None:
                    return label in _in if is_whitelist else label not in _in  # pylint: disable=unsupported-membership-test
                else:
                    return True
            else:
                if _out is not None:
                    return label in _out if is_whitelist else label not in _out  # pylint: disable=unsupported-membership-test
                else:
                    return True

    def delete(self):
        _outbound = self.outbound.copy()  # make a copy to iterate over
        for edge in _outbound:
            edge.delete()

        _inbound = self.inbound.copy()  # make a copy to iterate over
        for edge in _inbound:
            edge.delete()

        Node.objects.remove(self)
        self.deleted = True

    def copy_references(self, other: Node):
        """Create new edges for the other node with
        the same label. Note that only the label is
        equal and no other attribute is copied

        Arguments:
            other {Node} -- The node to receive the new edges
        """
        for e in self.inbound:
            e.source.add_edge(e.label, other)

        for e in self.outbound:
            other.add_edge(e.label, e.destination)

    def move_references(self, other: Node, outbound=True, inbound=True):
        if inbound:
            for e in self.inbound.copy():
                e.destination = other

        if outbound:
            for e in self.outbound.copy():
                e.source = other

    def replace_with(self, other: Node, **kwargs):
        """Moves the references and deletes the node

        Arguments:
            other {Node} -- The node to replace this one
        """
        self.move_references(other, **kwargs)
        self.delete()

    def get_all_linked(self):
        all_nodes: Set[Node] = set()
        nodes_to_explore: Set[Node] = set([self])
        while len(nodes_to_explore) > 0:
            all_nodes.update(nodes_to_explore)
            new_nodes_to_explore: Set[Node] = set()
            for node_to_explore in nodes_to_explore:
                connected_nodes = node_to_explore.query.through_edge().results
                connected_nodes = cast(Set[Node], connected_nodes)
                new_nodes_to_explore.update([node for node in connected_nodes if node not in all_nodes])

            nodes_to_explore = new_nodes_to_explore
        return all_nodes

    def to_nx_graph(self, formatting=lambda x: x):
        all_nodes: Set[Node] = set()
        nodes_to_explore: Set[Node] = set([self])
        while len(nodes_to_explore) > 0:
            all_nodes.update(nodes_to_explore)
            new_nodes_to_explore: Set[Node] = set()
            for node_to_explore in nodes_to_explore:
                connected_nodes = node_to_explore.query.through_edge().results
                connected_nodes = cast(Set[Node], connected_nodes)
                new_nodes_to_explore.update([node for node in connected_nodes if node not in all_nodes])

            nodes_to_explore = new_nodes_to_explore

        links = []
        for node in all_nodes:
            for out in node.outbound:
                links.append((formatting(node), formatting(out.destination), {'label': out.label}))

        import networkx as nx
        g = nx.DiGraph()
        g.add_edges_from(links)

        node_attrs = {}
        for node in all_nodes:
            node_attrs[formatting(node)] = {
                'Label': node.label,
                'type': node.label,
                'description': str(node),
                'id': node.graph_id
            }

        nx.set_node_attributes(g, node_attrs)

        return g

    def to_image(self, destination: str, pydot: bool = False):
        g = self.to_nx_graph()
        if pydot:
            from networkx.drawing.nx_pydot import to_pydot
            pd = to_pydot(g)
            pd.write_png(destination)
        else:
            from networkx.drawing.nx_agraph import to_agraph
            a = to_agraph(g)
            # a.layout(prog='sfdp')
            a.draw(destination, prog='fdp', args='-Goverlap=false -GK=2')

    def to_gml(self, destination: str):
        g = self.to_nx_graph(formatting=lambda n: str(n).replace('\n', ' - '))
        from networkx import write_gml
        write_gml(g, destination)

    def to_3d_html(self, destination: str):
        import networkx as nx
        import math
        import plotly.graph_objects as go
        import plotly.offline
        import matplotlib.pyplot as plt

        def addEdge(start,
                    end,
                    edge_x,
                    edge_y,
                    lengthFrac=1,
                    arrowPos=None,
                    arrowLength=0.025,
                    arrowAngle=30,
                    dotSize=20):
            """
            @author: aransil
            https://github.com/redransil/plotly-dirgraph/blob/master/addEdge.py
            """
            # Get start and end cartesian coordinates
            x0, y0 = start
            x1, y1 = end

            # Incorporate the fraction of this segment covered by a dot into total reduction
            length = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
            dotSizeConversion = .0565 / 20  # length units per dot size
            convertedDotDiameter = dotSize * dotSizeConversion
            lengthFracReduction = convertedDotDiameter / length
            lengthFrac = lengthFrac - lengthFracReduction

            # If the line segment should not cover the entire distance, get actual start and end coords
            skipX = (x1 - x0) * (1 - lengthFrac)
            skipY = (y1 - y0) * (1 - lengthFrac)
            x0 = x0 + skipX / 2
            x1 = x1 - skipX / 2
            y0 = y0 + skipY / 2
            y1 = y1 - skipY / 2

            # Append line corresponding to the edge
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)  # Prevents a line being drawn from end of this edge to start of next edge
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

            # Draw arrow
            if not arrowPos == None:

                # Find the point of the arrow; assume is at end unless told middle
                pointx = x1
                pointy = y1
                eta = math.degrees(math.atan((x1 - x0) / (y1 - y0)))

                if arrowPos == 'middle' or arrowPos == 'mid':
                    pointx = x0 + (x1 - x0) / 2
                    pointy = y0 + (y1 - y0) / 2

                # Find the directions the arrows are pointing
                signx = (x1 - x0) / abs(x1 - x0)
                signy = (y1 - y0) / abs(y1 - y0)

                # Append first arrowhead
                dx = arrowLength * math.sin(math.radians(eta + arrowAngle))
                dy = arrowLength * math.cos(math.radians(eta + arrowAngle))
                edge_x.append(pointx)
                edge_x.append(pointx - signx**2 * signy * dx)
                edge_x.append(None)
                edge_y.append(pointy)
                edge_y.append(pointy - signx**2 * signy * dy)
                edge_y.append(None)

                # And second arrowhead
                dx = arrowLength * math.sin(math.radians(eta - arrowAngle))
                dy = arrowLength * math.cos(math.radians(eta - arrowAngle))
                edge_x.append(pointx)
                edge_x.append(pointx - signx**2 * signy * dx)
                edge_x.append(None)
                edge_y.append(pointy)
                edge_y.append(pointy - signx**2 * signy * dy)
                edge_y.append(None)

            return edge_x, edge_y

        nodeColor = 'Blue'
        nodeSize = 20
        lineWidth = 2
        lineColor = '#000000'
        g = self.to_nx_graph()
        pos = nx.layout.spring_layout(g, iterations=10**3, seed=1337)

        # tests:
        # nx.draw(g, pos=nx.spring_layout(g, iterations=10**6, seed=1337))
        # plt.draw()
        # plt.show()
        # end tests

        for node in g.nodes:
            g.nodes[node]['pos'] = [p for p in pos[node]]

        # Make list of nodes for plotly
        node_x = []
        node_y = []
        for node in g.nodes():
            x, y = g.nodes[node]['pos']
            node_x.append(x)
            node_y.append(y)

        # Make a list of edges for plotly, including line segments that result in arrowheads
        edge_x = []
        edge_y = []
        for edge in g.edges():
            # addEdge(start, end, edge_x, edge_y, lengthFrac=1, arrowPos = None, arrowLength=0.025, arrowAngle = 30, dotSize=20)
            start = g.nodes[edge[0]]['pos']
            end = g.nodes[edge[1]]['pos']
            edge_x, edge_y = addEdge(start=start,
                                     end=end,
                                     edge_x=edge_x,
                                     edge_y=edge_y,
                                     lengthFrac=.99,
                                     arrowPos='end',
                                     arrowLength=.04,
                                     arrowAngle=30,
                                     dotSize=nodeSize)

        edge_trace = go.Scatter(x=edge_x,
                                y=edge_y,
                                line=dict(width=lineWidth, color=lineColor),
                                hoverinfo='none',
                                mode='lines')

        node_trace = go.Scatter(x=node_x,
                                y=node_y,
                                mode='markers',
                                hoverinfo='text',
                                marker=dict(showscale=False, color=nodeColor, size=nodeSize))

        fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(showlegend=False,
                                         hovermode='closest',
                                         margin=dict(b=20, l=5, r=5, t=40),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

        # Note: if you don't use fixed ratio axes, the arrows won't be symmetrical
        fig.update_layout(yaxis=dict(scaleanchor="x", scaleratio=1), plot_bgcolor='rgb(255,255,255)')
        plotly.offline.plot(fig)

    def to_interactive_html(self, destination: str):
        from pyvis.network import Network  # type: ignore
        G = Network(height='100%', width='100%', directed=True, bgcolor='#000000')
        G.toggle_stabilization(False)

        all_nodes: Set[Node] = set()
        nodes_to_explore: Set[Node] = set([self])
        while len(nodes_to_explore) > 0:
            all_nodes.update(nodes_to_explore)
            new_nodes_to_explore: Set[Node] = set()
            for node_to_explore in nodes_to_explore:
                connected_nodes = node_to_explore.query.through_edge().results
                connected_nodes = cast(Set[Node], connected_nodes)
                new_nodes_to_explore.update([node for node in connected_nodes if node not in all_nodes])

            nodes_to_explore = new_nodes_to_explore

        for node in all_nodes:
            G.add_node(node.graph_id, label=str(node))

        for node in all_nodes:
            for out in node.outbound:
                G.add_edge(node.graph_id, out.destination.graph_id)
                # links.append((node, out.destination, {'label': out.label}))

        G.show(destination)
        # G.enable_physics(True)
        # G.show(destination + '.phys.html')


class NotifyingSet(set, Generic[T]):
    def __init__(self, remove_callback: Callable[[T], None], add_callback: Callable[[T], None], *args, **kwargs):
        self.remove_callback = remove_callback
        self.add_callback = add_callback

    def add(self, item: T):
        self.add_callback(item)
        super().add(item)

    def remove(self, item: T):
        self.remove_callback(item)
        super().remove(item)


class NodeList(Generic[TN], Node):
    _base_label = 'node_list_item_'

    class LabelNotIndex(Exception):
        pass

    class NoOutboundEdgeAllowed(Exception):
        pass

    def __init__(self, contents: Optional[Iterable[TN]] = None):
        super().__init__()
        self.outbound = NotifyingSet(remove_callback=self._on_item_remove, add_callback=self._on_item_add)
        self._adding: Optional[str] = None
        self._fix_ordering = True
        if contents:
            for e in contents:
                self.append(e)

    @classmethod
    def get_label_by_index(cls, index: int):
        return cls._base_label + str(index)

    @classmethod
    def get_index_from_label(cls, label: str):
        if label.startswith(cls._base_label):
            return int(label[len(cls._base_label):])
        else:
            raise cls.LabelNotIndex('Label is not index')

    @classmethod
    def get_lists_that_contains(cls, item: Node, check_type: bool = False):
        q = item.query.through_incoming_edge(lambda x: x.startswith(cls._base_label))
        results = q.results
        if check_type:
            return {el.ensure_type(NodeList) for el in results}
        else:
            return cast(Set[NodeList], results)

    @property
    def items(self) -> List[TN]:
        edges = sorted(self.query.to_outgoing_edge().results, key=lambda x: self.get_index_from_label(x.label))
        edges = cast(List[Edge], edges)
        return [e.destination for e in edges]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index) -> TN:
        return self.items[index]

    def __contains__(self, element) -> bool:
        return element in self.items

    def __iter__(self) -> Iterator[TN]:
        return iter(self.items)

    def __setitem__(self, index: int, value: TN) -> None:
        check_and_raise(index < len(self))

        # remove old item at index without reordering
        self._fix_ordering = False
        self.query.to_outgoing_edge(self.get_label_by_index(index)).one.delete()
        self._fix_ordering = True

        # add new item at index
        new_label = self.get_label_by_index(index)
        self._adding = new_label
        self.add_edge(new_label, to=value)
        self._adding = None

    def append(self, item: TN) -> None:
        new_item_index = len(self)
        new_label = self.get_label_by_index(new_item_index)

        self._adding = new_label
        self.add_edge(edge_label=new_label, to=item)
        self._adding = None

    def insert(self, index: int, item: Union[TN, Sequence[TN]]) -> None:
        check_and_raise(index <= len(self))
        if isinstance(item, Sequence):
            for i in reversed(item):
                self.insert(index, i)
        else:
            # get edges from index onwards
            all_item_edges = self.query.to_outgoing_edge().results
            edges = filter(lambda e: self.get_index_from_label(e.label) >= index, all_item_edges)

            # shift them by one
            for edge in edges:
                edge.label = self.get_label_by_index(self.get_index_from_label(edge.label) + 1)

            # add new item at index
            new_label = self.get_label_by_index(index)
            self._adding = new_label
            self.add_edge(new_label, item)
            self._adding = None

    def index(self, item: TN) -> int:
        for i in range(len(self)):
            if self[i] == item:
                return i
        raise IndexError

    def remove(self, item: TN) -> None:
        self.pop(self.index(item))

    def pop(self, index: int):
        self.query.to_outgoing_edge(self.get_label_by_index(index)).one.delete()
        # self._retract_on_index(index) # will be called from _on_item_remove

    def _on_item_remove(self, edge_to_remove: Edge):
        if self._fix_ordering:
            try:
                idx = self.get_index_from_label(edge_to_remove.label)
                self._retract_on_index(idx)
            except self.LabelNotIndex:
                # label is not index
                pass

    def _on_item_add(self, edge_to_add: Edge):
        if self._adding is None:
            raise self.NoOutboundEdgeAllowed('Manual outbound edge setting is not allowed')

        elif self._adding == edge_to_add.label:
            pass

        else:
            raise Exception('Appending edge does not match expected value')

    def _retract_on_index(self, index):
        edges = [e for e in self.query.to_outgoing_edge().results if int(e.label) > index]
        for edge in edges:
            edge.label = str(int(edge.label) - 1)


class QuantityError(Exception):
    pass


class Query:
    def __init__(self, start: Optional[Union[GraphObject, Iterable[GraphObject]]], lazy: bool = False):
        self.lazy = lazy
        if self.lazy:
            if start is not None:
                raise Exception("Start must be None when using lazy query")
            else:
                self._calls = []

        else:
            if isinstance(start, GraphObject):
                self.results = [start]

            elif isinstance(start, Iterable):
                self.results = start
            elif start is None:
                raise Exception("Start cannot be None when using non-lazy query. Please include a start or use one of the static methods from_nodes() or from_node_label(label:str)")
            else:
                raise Exception('Unknown type')

    @property
    def results(self):
        check_and_raise(not self.lazy)
        return self._results

    @results.setter
    def results(self, value: Iterable[GraphObject]):
        check_and_raise(not self.lazy)
        self._results = set(value)

    def results_lazy(self, start: Iterable[GraphObject]):
        '''Gets the lazy query result given a start point'''
        if isinstance(start, GraphObject):
                start = [start]
        results = start
        for call in self._calls:
            results = call(results)
        return results

    def execute(self, start: Iterable[GraphObject]) -> 'Query':
        '''Creates a non-lazy query from a lazy query'''
        results = self.results_lazy(start)
        return Query(start=results)

    @property
    def one(self) -> GraphObject:
        quantity = len(self.results)
        check_and_raise(quantity == 1, QuantityError(f'Expected to have one object. Has {quantity}'))
        return next(iter(self.results))

    @property
    def one_or_none(self) -> Optional[GraphObject]:
        quantity = len(self.results)
        check_and_raise(quantity <= 1, QuantityError(f'Expected to have one or no objects. Has {quantity}'))
        if self.results:
            return next(iter(self.results))
        else:
            return None

    def has_any(self):
        return len(self.results) > 0

    def ensure_zero(self):
        quantity = len(self.results)
        check_and_raise(quantity == 0, QuantityError(f'Expected to have no objects. Has {quantity}'))

    @staticmethod
    def from_nodes():
        return Query(start=Node.objects)

    @staticmethod
    def from_node_label(label: str):
        return Query.from_nodes().filter_by_label(label)

    # Traversals

    def to_outgoing_edge(self, edge_label: Union[str, None] = None) -> 'Query':
        def action(results):
            new_results = []
            for node in results:
                if isinstance(node, Node):
                    new_results += [e for e in node.outbound if e.label == edge_label or edge_label is None]
            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def to_incoming_edge(self, edge_label: Union[str, None] = None) -> 'Query':
        def action(results):
            new_results = []
            for node in results:
                if isinstance(node, Node):
                    new_results += [e for e in node.inbound if e.label == edge_label or edge_label is None]
            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def to_edge(self, edge_label: Union[str, None] = None) -> 'Query':
        def action(results):
            new_results = []
            for node in results:
                if isinstance(node, Node):
                    new_results += [e for e in node.outbound if e.label == edge_label or edge_label is None]
                    new_results += [e for e in node.inbound if e.label == edge_label or edge_label is None]

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def to_source_node(self) -> 'Query':
        def action(results):
            new_results = []
            for edge in results:
                if isinstance(edge, Edge):
                    new_results.append(edge.source)

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def to_destination_node(self) -> 'Query':
        def action(results):
            new_results = []
            for edge in results:
                if isinstance(edge, Edge):
                    new_results.append(edge.destination)

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def through_outgoing_edge(self, edge_label: Union[str, None] = None) -> 'Query':
        def action(results):
            new_results = []
            for node in results:
                if isinstance(node, Node):
                    new_results += [e.destination for e in node.outbound if e.label == edge_label or edge_label is None]

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def through_incoming_edge(self, edge_label: Optional[Union[str, Callable[[str], bool]]] = None) -> 'Query':
        if edge_label is None:
            filter_func: Callable[[str], bool] = lambda x: True

        elif isinstance(edge_label, Callable):
            filter_func = cast(Callable[[str], bool], edge_label)

        else:
            filter_func = lambda x: x == edge_label

        _filter_func: Callable[[Edge], bool] = lambda n: filter_func(n.label)

        def action(results):
            new_results = []
            for node in results:
                if isinstance(node, Node):
                    # it's split in two lines to be easier to see the results in the debugger
                    filtered_list = [e.source for e in filter(_filter_func, node.inbound)]
                    new_results += filtered_list

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def through_edge(self, edge_label: Union[str, None] = None) -> 'Query':
        def action(results):
            new_results = []
            for node in results:
                if isinstance(node, Node):
                    new_results += [e.source for e in node.inbound if e.label == edge_label or edge_label is None]
                    new_results += [e.destination for e in node.outbound if e.label == edge_label or edge_label is None]

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    # Filters

    def filter_by_relation(self, relation: Query) -> 'Query':
        check_and_raise(relation.lazy)

        def action(results: Iterable[GraphObject]):
            new_results = []

            for element in results:
                if relation.execute(element).has_any():
                    new_results.append(element)

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def filter_by_label(self, *labels: Union[str, GraphObject, Type[GraphObject]]) -> 'Query':
        labelss = labels  # weird pyright error

        def action(results):
            new_results = []
            for label in labelss:  # weird pyright error
                if isinstance(label, GraphObject):
                    label = label.label
                elif (not isinstance(label, str)) and issubclass(label, GraphObject):
                    label = label.__name__

                new_results += [r for r in results if r.label == label]

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self

    def filter_by_property(self, **kwproperties) -> 'Query':
        def action(results):
            new_results = []
            for result in results:
                can_add = True
                for key, value in kwproperties.items():
                    if hasattr(result, key):
                        if getattr(result, key) == value:
                            # great, let's go on
                            pass
                        else:
                            can_add = False
                            break
                    else:
                        can_add = False
                        break
                if can_add:
                    new_results.append(result)

            return new_results

        if self.lazy:
            self._calls.append(action)
        else:
            self.results = action(self.results)
        return self


def relation():
    return Query(start=None, lazy=True)


ActionType = Callable[[Iterable[GraphObject]], Iterable[GraphObject]]

# def filter_decorator(func: Callable[..., Tuple[Type[Query[TG]], ActionType]]) -> Callable[..., Query[TG]]:
#     def wrapper(self, *args, **kwargs) -> Query[TG]:
#         q_type, action = func(self, *args, **kwargs)
#         if self.lazy:
#             q = q_type(start=None, lazy=True)
#             q._calls = self._calls.copy()
#             q._calls.append(action)
#             return q
#         else:
#             q = q_type(start=action(self.results))
#             return q

#     return wrapper
