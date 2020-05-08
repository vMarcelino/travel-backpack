from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Union, Iterable


class GraphObject:
    objects = []
    object_count = 0

    def __init__(self, object_base_class: 'GraphObject', label=None):
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
        setattr(self, 'delete', start_delete) # in this form to avoid linting errors

    def __hash__(self):
        return hash(self.graph_id)

    def __str__(self):
        return f'{self.class_id}\n{self.label}'

    def __repr__(self):
        return f'{self.label}'

    def delete(self):
        raise NotImplementedError


class Edge(GraphObject):
    def __init__(self, label: str, source: 'Node', destination: 'Node'):
        super().__init__(Edge, label=label)

        self.source = source
        self.destination = destination

    def delete(self):
        self.source.outbound.remove(self)
        self.destination.inbound.remove(self)
        self.source = None
        self.destination = None
        Edge.objects.remove(self)
        self.deleted = True


class Node(GraphObject):
    # whitelist or blacklist
    __list_mode__ = 'whitelist'

    # if defined, all listed here will be in the list,
    # regardless if it's inbound or outbound.
    # if None, disconsider this check
    __edge_list__: Union[None, List[str]] = None

    __inbound_edge_list__: Union[None, List[str]] = None
    __outbound_edge_list__: Union[None, List[str]] = None

    def __init__(self, label: str = None):
        super().__init__(Node, label=label)

        self.inbound: List[Edge] = []
        self.outbound: List[Edge] = []

    def add_edge(self, edge_label: Union[str, Edge, type], to: Node, two_way=False):
        if isinstance(edge_label, Edge):
            edge_label = edge_label.label

        elif isinstance(edge_label, type):
            edge_label = edge_label.__name__

        if self._is_edge_allowed(edge_label, inbound=False) and to._is_edge_allowed(edge_label, inbound=True):
            edge = Edge(source=self, destination=to, label=edge_label)
            self.outbound.append(edge)
            to.inbound.append(edge)
            if two_way:
                to.add_edge(edge_label=edge_label, to=self, two_way=False)
        else:
            raise Exception("Edge label not allowed: " + edge_label)

    def _is_edge_allowed(self, label: str, inbound: True):
        lm = self.__list_mode__
        wl = lm == 'whitelist'
        a = self.__edge_list__
        i = self.__inbound_edge_list__
        o = self.__outbound_edge_list__

        if a is not None:
            if inbound:
                if i is not None:
                    return label in a + i if wl else label not in a + i
                else:
                    return label in a if wl else label not in a
            else:
                if o is not None:
                    return label in a + o if wl else label not in a + o
                else:
                    return label in a if wl else label not in a
        else:
            if inbound:
                if i is not None:
                    return label in i if wl else label not in i
                else:
                    return True
            else:
                if o is not None:
                    return label in o if wl else label not in o
                else:
                    return True

    @property
    def query(self):
        return Query(start=self)

    def delete(self):
        _outbound = self.outbound[:]  # make a copy to iterate over
        for edge in _outbound:
            edge.delete()

        _inbound = self.inbound[:]  # make a copy to iterate over
        for edge in _inbound:
            edge.delete()

        Node.objects.remove(self)
        self.deleted = True

    def copy_references(self, other: 'Node'):
        for e in self.inbound:
            e.source.add_edge(e.label, other)

        for e in self.outbound:
            other.add_edge(e.label, e.destination)

    @staticmethod
    def to_nx_graph():
        all_nodes: List[Node] = []
        for v in Node.objects:
            all_nodes.append(v)

        links = []
        for node in all_nodes:
            for out in node.outbound:
                links.append((node, out.destination, {'label': out.label}))

        import networkx as nx
        g = nx.DiGraph()
        g.add_edges_from(links)
        return g
    @staticmethod
    def to_image(destination:str):
        g = Node.to_nx_graph()
        from networkx.drawing.nx_pydot import to_pydot
        pd = to_pydot(g)
        pd.write_png(destination)



class NotifyingList(list):
    def __init__(self, remove_callback, *args, **kwargs):
        self.remove_callback = remove_callback

    def remove(self, item):
        self.remove_callback(item)
        super().remove(item)


class NodeList(Node):
    def __init__(self, contents: Iterable = None):
        super().__init__()
        self.outbound = NotifyingList(self.on_item_remove)
        if contents:
            for e in contents:
                self.append(e)

    @property
    def items(self) -> List[Node]:
        edges = sorted(self.query.to_outgoing_edge().results, key=lambda x: int(x.label))
        return [e.destination for e in edges]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index) -> Node:
        return self.items[index]

    def __contains__(self, element):
        return element in self.items

    def __iter__(self):
        return iter(self.items)

    def __setitem__(self, index, value):
        assert index < len(self)
        self.query.to_outgoing_edge(str(index)).one.delete()
        self.add_edge(str(index), to=value)

    def append(self, item):
        new_item_index = str(len(self))
        self.add_edge(edge_label=new_item_index, to=item)

    def insert(self, index, item):
        assert index <= len(self)
        if isinstance(item, (list, tuple)):
            for i in reversed(item):
                self.insert(index, i)
        else:
            edges = [e for e in self.query.to_outgoing_edge().results if int(e.label) >= index]
            for edge in edges:
                edge.label = str(int(edge.label) + 1)

            self.add_edge(str(index), item)

    def index(self, item):
        for i in range(len(self)):
            if self[i] == item:
                return i

    def remove(self, item):
        self.pop(self.index(item))

    def _retract_on_index(self, index):
        edges = [e for e in self.query.to_outgoing_edge().results if int(e.label) > index]
        for edge in edges:
            edge.label = str(int(edge.label) - 1)

    def pop(self, index):
        self.query.to_outgoing_edge(str(index)).one.delete()
        self._retract_on_index(index)

    def on_item_remove(self, edge_to_remove: Edge):
        idx = int(edge_to_remove.label)
        self._retract_on_index(idx)


class Query:
    def __init__(self, start: Union[GraphObject, List[GraphObject], None], lazy=False):
        self.lazy = lazy
        if self.lazy:
            if start is not None:
                Exception("Start must be None when using lazy query")
            else:
                self._calls = []

        else:
            if isinstance(start, GraphObject):
                self.results = [start]

            elif isinstance(start, List):
                self.results = start

            else:
                raise Exception('Unknown type')

    @property
    def results(self):
        assert not self.lazy
        return self._results

    @results.setter
    def results(self, value):
        assert not self.lazy
        self._results = value

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
        assert len(self.results) == 1
        return self.results[0]

    @property
    def one_or_none(self) -> Union[GraphObject, None]:
        assert len(self.results) <= 1
        if self.results:
            return self.results[0]
        else:
            return None

    def has_any(self):
        return len(self.results) > 0

    def ensure_zero(self):
        assert len(self.results) == 0

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
        assert relation.lazy

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

    def filter_by_label(self, *labels: Union[str, GraphObject, type]) -> 'Query':
        def action(results):
            new_results = []
            for label in labels:
                if isinstance(label, GraphObject):
                    label = label.label
                elif isinstance(label, type):
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
                        if getattr(result.key) == value:
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