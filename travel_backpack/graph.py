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
    Literal,
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
            label = instance_class.__name__

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


class Edge(GraphObject):
    def __init__(self, label: str, source: 'Node', destination: 'Node'):
        super().__init__(Edge, label=label)
        self._source = None
        self._destination = None

        self.source = source
        self.destination = destination

    @property
    def source(self) -> 'Node':
        return ensure_type(self._source, Node)

    @source.setter
    def source(self, new_source: 'Node'):
        if new_source._is_edge_allowed(self.label, inbound=False):
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
        if new_destination._is_edge_allowed(self.label, inbound=True):
            if self._destination is not None:
                old_destination = self.destination
                old_destination.inbound.remove(self)
            new_destination.inbound.add(self)
            self._destination = new_destination
        else:
            raise EdgeLabelNotAllowed(f'Inbound edge label is not allowed on this node: {self.label}')

    def delete(self):
        self.source.outbound.remove(self)
        self.destination.inbound.remove(self)
        Edge.objects.remove(self)
        self.deleted = True
        # self.source = None
        # self.destination = None


class Node(GraphObject):
    # whitelist or blacklist
    __list_mode__: Literal['whitelist', 'blacklist'] = 'whitelist'

    # if defined, all listed here will be in the list,
    # regardless if it's inbound or outbound.
    # if None, disconsider this check
    __edge_list__: Optional[List[str]] = None

    __inbound_edge_list__: Optional[List[str]] = None
    __outbound_edge_list__: Optional[List[str]] = None

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
        _all = self.__edge_list__
        _in = self.__inbound_edge_list__
        _out = self.__outbound_edge_list__

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

    @property
    def query(self):
        return Query(start=self)

    def delete(self):
        _outbound = self.outbound.copy()  # make a copy to iterate over
        for edge in _outbound:
            edge.delete()

        _inbound = self.inbound.copy()  # make a copy to iterate over
        for edge in _inbound:
            edge.delete()

        Node.objects.remove(self)
        self.deleted = True

    def copy_references(self, other: 'Node'):
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

    def move_references(self, other: 'Node'):
        for e in self.inbound:
            e.destination = other

        for e in self.outbound:
            e.source = other

    def replace_with(self, other: 'Node'):
        """Moves the references and deletes the node

        Arguments:
            other {Node} -- The node to replace this one
        """
        self.move_references(other)
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

    def to_nx_graph(self):
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
                links.append((node, out.destination, {'label': out.label}))

        import networkx as nx
        g = nx.DiGraph()
        g.add_edges_from(links)
        return g

    def to_image(self, destination: str):
        g = self.to_nx_graph()
        from networkx.drawing.nx_pydot import to_pydot
        pd = to_pydot(g)
        pd.write_png(destination)

    def to_interactive_html(self, destination: str):
        from pyvis.network import Network
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
                Exception("Start must be None when using lazy query")
            else:
                self._calls = []

        else:
            if isinstance(start, GraphObject):
                self.results = [start]

            elif isinstance(start, Iterable):
                self.results = start

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

    def results_lazy(self, start):
        results = start
        for call in self._calls:
            results = call(results)
        return results

    def execute(self, start) -> 'Query':
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

    def through_incoming_edge(self, edge_label: Union[str, None] = None) -> 'Query':
        def action(results):
            new_results = []
            for node in results:
                if isinstance(node, Node):
                    new_results += [e.source for e in node.inbound if e.label == edge_label or edge_label is None]

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

        def action(results):
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
