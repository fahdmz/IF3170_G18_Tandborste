import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from src.models.decision_tree import DecisionTree, TreeNode


def _plot(model, path, **kwargs):
    renderer = importlib.import_module("src.visualization.decision_tree_graph")
    renderer.plot_decision_tree(model, path, **kwargs)


def _export(model, path, **kwargs):
    if path.suffix == ".dot":
        model.export_graph(path, **kwargs)
    else:
        _plot(model, path, **kwargs)


@pytest.fixture
def mixed_tree():
    def leaf(counts, depth):
        return TreeNode(True, int(np.argmax(counts)), np.array(counts),
                        n_samples=sum(counts), depth=depth)

    left = TreeNode(False, 0, np.array([3.25, 3.25]), feature_index=1,
                    is_categorical=True, n_samples=6.5, depth=1,
                    children={"C": leaf([1.0, 0.0], 2),
                              'A"\\\nroom': leaf([2.25, 0.25], 2),
                              "B": leaf([0.0, 3.0], 2)})
    right = TreeNode(False, 0, np.array([3.0, 3.0]), feature_index=0,
                     threshold=5.5, n_samples=6.0, depth=1,
                     children={"gt": leaf([0.0, 3.0], 2), "le": leaf([3.0, 0.0], 2)})
    model = DecisionTree(categorical_features=[1])
    model.tree_ = TreeNode(False, 0, np.array([6.25, 6.25]), feature_index=0,
                           threshold=2.5, n_samples=12.5,
                           children={"gt": right, "le": left})
    model.classes_ = np.array([0, 1])
    model.n_features_in_ = 2
    model._categorical_features_ = frozenset({1})
    return model


def _read_dot(path):
    source = path.read_text(encoding="utf-8")
    quoted = r'"(?:[^"\\]|\\.)*"'
    nodes = {int(index): json.loads(label) for index, label in re.findall(
        rf'^\s*n(\d+) \[label=({quoted})\];$', source, flags=re.MULTILINE)}
    edges = [(int(parent), int(child), json.loads(label))
             for parent, child, label in re.findall(
                 rf'^\s*n(\d+) -> n(\d+) \[label=({quoted})\];$', source, flags=re.MULTILINE)]
    return source, nodes, edges


def test_dot_has_numeric_category_edges_and_escaped_utf8_labels(tmp_path, mixed_tree):
    path = tmp_path / "nested" / "tree.dot"
    names = ['price"\\\nIDR', "kategori kamar é"]
    mixed_tree.export_graph(path, feature_names=names)
    source, nodes, edges = _read_dot(path)
    assert source.startswith("digraph DecisionTree {") and source.rstrip().endswith("}")
    assert len(nodes) == 8
    assert edges == [(0, 1, "<="), (0, 2, ">"), (1, 3, 'A"\\\nroom'),
                     (1, 4, "B"), (1, 5, "C"), (2, 6, "<="), (2, 7, ">")]
    assert names[0] + " <= 2.5" in nodes[0]
    assert names[1] in nodes[1]
    assert "mass = 12.5" in nodes[0] and "counts = [6.25, 6.25]" in nodes[0]
    assert "prediction = 0" in nodes[3] and "counts = [2.25, 0.25]" in nodes[3]


@pytest.mark.parametrize("top_n, count, edge_pairs", [
    (1, 1, []), (2, 2, [(0, 1)]),
    (4, 4, [(0, 1), (0, 2), (1, 3)]),
    (None, 8, [(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7)]),
    (99, 8, [(0, 1), (0, 2), (1, 3), (1, 4), (1, 5), (2, 6), (2, 7)]),
])
def test_dot_top_n_counts_actual_bfs_nodes_including_root(tmp_path, mixed_tree, top_n, count, edge_pairs):
    path = tmp_path / "tree.dot"
    mixed_tree.export_graph(path, top_n=top_n)
    _, nodes, edges = _read_dot(path)
    assert sorted(nodes) == list(range(count))
    assert [(parent, child) for parent, child, _ in edges] == edge_pairs
    assert "x0 <= 2.5" in nodes[0]
    if top_n == 1:
        assert "2 branches hidden" in nodes[0]
    elif top_n == 2:
        assert "1 branches hidden" in nodes[0] and "3 branches hidden" in nodes[1]
    elif top_n == 4:
        assert "2 branches hidden" in nodes[1] and "2 branches hidden" in nodes[2]
    else:
        assert all("hidden" not in label for label in nodes.values())


@pytest.mark.parametrize("extension", [".dot", ".png"])
@pytest.mark.parametrize("top_n", [0, -1, 1.5, True, "3"])
def test_invalid_top_n_is_rejected_without_creating_file(tmp_path, mixed_tree, extension, top_n):
    path = tmp_path / ("tree" + extension)
    with pytest.raises(ValueError, match="top_n"):
        _export(mixed_tree, path, top_n=top_n)
    assert not path.exists()


@pytest.mark.parametrize("extension", [".dot", ".png"])
@pytest.mark.parametrize("names", [[], ["only_one"], ["a", "b", "extra"]])
def test_feature_name_count_must_match_fitted_schema(tmp_path, mixed_tree, extension, names):
    with pytest.raises(ValueError, match="feature_names"):
        _export(mixed_tree, tmp_path / ("tree" + extension), feature_names=names)


@pytest.mark.parametrize("extension", [".dot", ".png"])
def test_export_requires_fitted_model(tmp_path, extension):
    with pytest.raises(RuntimeError, match="fit"):
        _export(DecisionTree(), tmp_path / ("tree" + extension))


@pytest.mark.parametrize("method, suffix", [("dot", ".png"), ("dot", ".txt"),
                                            ("png", ".dot"), ("png", "")])
def test_wrong_extension_has_clear_error(tmp_path, mixed_tree, method, suffix):
    path = tmp_path / ("wrong" + suffix)
    with pytest.raises(ValueError, match="\\." + method):
        if method == "dot":
            mixed_tree.export_graph(path)
        else:
            _plot(mixed_tree, path)
    assert not path.exists()


def test_root_only_fitted_tree_exports_both_formats(tmp_path):
    model = DecisionTree().fit([[0], [1], [2]], [1, 1, 1])
    path = tmp_path / "leaf.dot"
    model.export_graph(path, top_n=1)
    _, nodes, edges = _read_dot(path)
    assert len(nodes) == 1 and not edges
    assert "prediction = 1" in nodes[0] and "counts = [0, 3]" in nodes[0]
    assert "hidden" not in nodes[0]
    _plot(model, tmp_path / "leaf.png", top_n=1)
    assert (tmp_path / "leaf.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("top_n, count, edge_count", [(1, 1, 0), (4, 4, 3), (99, 8, 7)])
def test_png_renders_only_selected_nodes_and_edges_headlessly(tmp_path, mixed_tree, monkeypatch,
                                                           top_n, count, edge_count):
    from matplotlib.figure import Figure
    from matplotlib import pyplot as plt

    saved = {}
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        axis = figure.axes[0]
        saved["nodes"] = [text.get_text() for text in axis.texts if "\nmass = " in text.get_text()]
        saved["edges"] = len(axis.lines)
        saved["figure"] = figure
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    before = plt.get_fignums()
    path = tmp_path / "nested" / "tree.png"
    _plot(mixed_tree, path, top_n=top_n, feature_names=['price"\\', "$room_type$"])
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") and path.stat().st_size > 1000
    assert len(saved["nodes"]) == count and saved["edges"] == edge_count
    assert 'price"\\ <= 2.5' in saved["nodes"][0]
    if top_n == 1:
        assert "2 branches hidden" in saved["nodes"][0]
    else:
        assert "$room_type$ (categorical)" in saved["nodes"][1]
    assert plt.get_fignums() == before
    assert not saved["figure"].axes


def test_exports_are_deterministic_and_do_not_mutate_tree(tmp_path, mixed_tree):
    queries = np.array([[1, "B"], [7, "C"], [None, "new"], [1, None]], dtype=object)
    labels, probabilities = mixed_tree.predict(queries), mixed_tree.predict_proba(queries)
    depth, leaves = mixed_tree.get_depth(), mixed_tree.get_n_leaves()

    def snapshot():
        result, pending = [], [mixed_tree.tree_]
        while pending:
            node = pending.pop()
            result.append((id(node), node.is_leaf, node.prediction, node.feature_index,
                           node.threshold, node.is_categorical, node.n_samples, node.impurity,
                           node.gain, node.depth, tuple(node.class_distribution),
                           tuple((key, id(child)) for key, child in node.children.items())))
            pending.extend(node.children.values())
        return result

    state = snapshot()
    for top_n in (1, 4, None, 99):
        first, second = tmp_path / "first.dot", tmp_path / "second.dot"
        mixed_tree.export_graph(first, top_n=top_n)
        mixed_tree.export_graph(second, top_n=top_n)
        assert first.read_bytes() == second.read_bytes()
        first_png, second_png = tmp_path / "first.png", tmp_path / "second.png"
        _plot(mixed_tree, first_png, top_n=top_n)
        _plot(mixed_tree, second_png, top_n=top_n)
        assert first_png.read_bytes() == second_png.read_bytes()
        assert snapshot() == state
        assert (mixed_tree.get_depth(), mixed_tree.get_n_leaves()) == (depth, leaves)
        np.testing.assert_array_equal(mixed_tree.predict(queries), labels)
        np.testing.assert_array_equal(mixed_tree.predict_proba(queries), probabilities)


def test_model_fit_predict_and_dot_work_without_renderer_or_optional_libraries(tmp_path):
    script = textwrap.dedent('''
        import importlib.abc
        import sys

        class BlockOptional(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {"matplotlib", "pandas", "sklearn"} or fullname.startswith("src.visualization"):
                    raise AssertionError("Forbidden model dependency: " + fullname)

        sys.meta_path.insert(0, BlockOptional())
        from src.models.decision_tree import DecisionTree
        model = DecisionTree().fit([[0]] * 10 + [[1]] * 10, [0] * 10 + [1] * 10)
        assert model.predict([[0], [1]]).tolist() == [0, 1]
        model.export_graph(sys.argv[1], top_n=2)
    ''')
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path / "plain.dot")],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    _, nodes, edges = _read_dot(tmp_path / "plain.dot")
    assert len(nodes) == 2 and len(edges) == 1
