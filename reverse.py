import string
from collections import Counter

import networkx
import plotly

from obj import H0_START, H1_START, H2_START, H3_START, H4_START, BitList, str_bits_to_list, chunks, \
    list_to_int, char_to_binary, sha1 as simple_sha1
import pushers.presumptive as ppr
from tree_bit.base import TreeBitAtom, UInt32Tree, registry, TreeBit, TreeBitNOT, TreeBitOperator, init_registry, \
    TreeBitOR, TreeBitAND, TreeBitXOR
from tree_bit.dnf import get_sdnf_for_bit, minimize_dnf, str_dnf
from tree_bit.tools import extract_base_bits

OperationsCounter = dict[tuple, int]
real_bits_scan: str = ''


h0_start_tree: UInt32Tree
h1_start_tree: UInt32Tree
h2_start_tree: UInt32Tree
h3_start_tree: UInt32Tree
h4_start_tree: UInt32Tree


def init_sha1_state(copy: bool = False):
    global h0_start_tree, h1_start_tree, h2_start_tree, h3_start_tree, h4_start_tree
    init_registry()

    h0_start_tree = UInt32Tree.from_int(H0_START, 'H0')
    h1_start_tree = UInt32Tree.from_int(H1_START, 'H1')
    h2_start_tree = UInt32Tree.from_int(H2_START, 'H2')
    h3_start_tree = UInt32Tree.from_int(H3_START, 'H3')
    h4_start_tree = UInt32Tree.from_int(H4_START, 'H4')


def calc_symbols_mask(used_symbols: list[str]):
    used_symbols = list(set(used_symbols))
    counter_mask = [0] * 8
    for char in used_symbols:
        if len(char) != 1:
            raise Exception('Bad used_symbols!')

        for index, bit in enumerate(str_bits_to_list(char_to_binary(char))):
            counter_mask[index] += bit

    probability_mask = [counted / len(used_symbols) for counted in counter_mask]
    return probability_mask


def sha1(data: str):
    global real_bits_scan
    init_sha1_state()

    bytes_: BitList = []
    for char in data:
        bytes_ += str_bits_to_list(char_to_binary(char))
    w, *h = algo(bytes_)
    result_bits = [bit for word in w[:16] for bit in word.bits]
    real_bits_scan = ''.join(str(int(bit.value)) if bit.resolved else 'X' for bit in result_bits)

    # sha_bits = [bit for hx in h for bit in hx.bits]
    # return scan_bits(sha_bits)
    h0 = h[0].to_int()
    h1 = h[1].to_int()
    h2 = h[2].to_int()
    h3 = h[3].to_int()
    h4 = h[4].to_int()
    return '%08x%08x%08x%08x%08x' % (h0, h1, h2, h3, h4)


def sha1_rev(sha: str, limit_length: int, used_symbols: list[str]):
    init_sha1_state()
    probability_mask = calc_symbols_mask(used_symbols)
    print(
        f'Probability mask: {probability_mask}\n'
        f'For charset: "{"".join(used_symbols)}"'
    )

    w, *predicted_h = forward_move(limit_length, probability_mask)

    # unpack hash bits
    hash_bits = []
    for h in predicted_h:
        for bit in h.bits:
            hash_bits.append(bit)

    print('Hash bits info: Base Count')
    for idx, hash_bit in enumerate(hash_bits, start=1):
        base_bits = extract_base_bits(hash_bit)
        counter = Counter(base_bits)
        filtered = filter(lambda kv: kv[1] > 1, counter.most_common())
        print(f'{idx}. ({len(counter)}, {counter.total()}) {dict(filtered)} {[b.name for b in base_bits]}')
        sdnf = get_sdnf_for_bit(hash_bit, True)

        # simple testing
        # sdnf = set(list(sdnf)[:5])

        print(str_dnf(sdnf))
        mdnf = minimize_dnf(sdnf)
        print(str_dnf(mdnf))
        # draw_bit_tree(hash_bit)
        # draw_graph_usages(base_bits)
        print()

    backward_move(sha, predicted_h, w)
    return None


def forward_move(limit_length: int, probability_mask: list[float]):
    # prepare from limit
    assert len(probability_mask) == 8
    bits: BitList = probability_mask * limit_length

    return algo(bits)


def algo(bits: BitList):
    from tree_bit.base import UINT_ZERO
    MAX_W = 80    # default
    # MAX_W = 0

    if MAX_W != 80:
        print('WARNING!!! MAX_W is not equal to 80! SHA1 algo is incorrect!')

    bits += [True]
    p_bits = bits.copy()
    # pad until length equals 448 mod 512
    if len(p_bits) % 512 > 448:
        p_bits += [False] * (512 - len(p_bits) % 512)
    if len(p_bits) % 512 < 448:
        p_bits += [False] * (448 - len(p_bits) % 512)
    # append the original length
    p_bits += str_bits_to_list(f'{len(bits) - 1:064b}')

    h0 = h0_start_tree
    h1 = h1_start_tree
    h2 = h2_start_tree
    h3 = h3_start_tree
    h4 = h4_start_tree

    # NOTE: only 1 chunk_512
    words: list[BitList] = chunks(p_bits, 32)
    w = [UINT_ZERO] * max(16, MAX_W)
    for n in range(0, 16):
        w[n] = UInt32Tree([
            TreeBit(bit, str(n * 32 + index))
            for index, bit in enumerate(words[n])
        ])
    for i in range(16, MAX_W):
        w[i] = (
            w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]
        ).rol(1)

    a = h0
    b = h1
    c = h2
    d = h3
    e = h4

    k1 = UInt32Tree.from_int(0x5A827999, 'K1')
    k2 = UInt32Tree.from_int(0x6ED9EBA1, 'K2')
    k3 = UInt32Tree.from_int(0x8F1BBCDC, 'K3')
    k4 = UInt32Tree.from_int(0xCA62C1D6, 'K4')

    # Main loop
    for i in range(0, MAX_W):
        if 0 <= i <= 19:
            f = (b & c) | ((~b) & d)
            k = k1
        elif 20 <= i <= 39:
            f = b ^ c ^ d
            k = k2
        elif 40 <= i <= 59:
            f = (b & c) | (b & d) | (c & d)
            k = k3
        else:
            f = b ^ c ^ d
            k = k4

        temp = a.rol(5) + f + e + k + w[i]
        e = d
        d = c
        c = b.rol(30)
        b = a
        a = temp

    h0 = h0 + a
    h1 = h1 + b
    h2 = h2 + c
    h3 = h3 + d
    h4 = h4 + e

    return w, h0, h1, h2, h3, h4


def backward_move(
        sha: str,
        predicted_h: tuple[UInt32Tree, ...],
        w: list[UInt32Tree],
):
    # prepare from sha
    h_end = tuple(
        UInt32Tree.from_int(int(sha[offset:offset + 8], 16), f'h{index}')
        for index, offset in enumerate(range(0, 40, 8))
    )
    result_bits = [bit for word in w[:16] for bit in word.bits]
    # scan_lines(predicted_h)
    draw_graph(
        predicted_h=predicted_h,
        usages_start_bits=result_bits,
    )
    # ppr.PredictsPusher().push_presumptive_predict(predicted_h, h_end, result_bits)


def scan_lines(predicted_h: tuple[UInt32Tree, ...]):
    next_line = []
    curr_line = [bit for h in predicted_h for bit in h.bits]
    print([bit.value for bit in curr_line])
    used_bits = set(curr_line)
    counter = 0
    while curr_line:
        used_prev = 0
        for bit in curr_line:
            if isinstance(bit, TreeBit):
                continue
            if isinstance(bit, TreeBitNOT):
                if bit.bit not in used_bits:
                    next_line.append(bit.bit)
                    used_bits.add(bit.bit)
                else:
                    if not bit.bit.resolved:
                        used_prev += 1
                continue
            if isinstance(bit, TreeBitOperator):
                if bit.a not in used_bits:
                    next_line.append(bit.a)
                    used_bits.add(bit.a)
                else:
                    if not bit.a.resolved:
                        used_prev += 1
                if bit.b not in used_bits:
                    next_line.append(bit.b)
                    used_bits.add(bit.b)
                else:
                    if not bit.b.resolved:
                        used_prev += 1
                continue
            raise Exception(f'Undefined TreeBit {type(bit)}')

        print(f'line {counter}: {len(curr_line)}, used on prev lines: {used_prev}')
        curr_line, next_line = next_line, []
        counter += 1

    print('registry size:', len(registry), 'used bits:', len(used_bits))


def draw_graph(
        predicted_h: tuple[UInt32Tree, ...] = None,
        predicted_h_bits: list[TreeBitAtom] = None,
        usages_start_bits: list[TreeBitAtom] = None,
        sizes=(60, 50)
):
    if predicted_h or predicted_h_bits:
        draw_graph_parents(predicted_h=predicted_h, predicted_h_bits=predicted_h_bits, sizes=sizes)
    if usages_start_bits:
        draw_graph_usages(start_bits=usages_start_bits, sizes=sizes)


def draw_graph_parents(
        predicted_h: tuple[UInt32Tree, ...] = None,
        predicted_h_bits: list[TreeBitAtom] = None,
        sizes=(60, 50)
):
    print('Start draw_graph_parents')
    graph = networkx.DiGraph()
    labels = {}

    node_color_map: dict[type[TreeBitAtom], str] = {
        TreeBit: "red",
        TreeBitNOT: 'blue',
        TreeBitOR: "orange",
        TreeBitAND: 'yellow', #"olive",
        TreeBitXOR: "green",
    }

    subset_tone = [
        "red",
        "orange",
        "olive",
        "green",
        "blue",
        "purple",
    ]

    next_line = []
    curr_line = []
    counter = 1

    if predicted_h:
        predicted_h_bits = []
        for h in predicted_h:
            for bit in h.bits:
                predicted_h_bits.append(bit)

    for bit in predicted_h_bits:
        curr_line.append(bit)
        color = node_color_map[type(bit)]
        graph.add_node(bit.key, subset=counter, color=color)
        labels[bit.key] = bit.label

    used_bits = set(curr_line)
    while curr_line and counter:
        counter += 1

        used_prev = 0
        for bit in curr_line:
            if isinstance(bit, TreeBit):
                continue

            for parent_bit in bit.parents:
                color = node_color_map[type(parent_bit)]
                if parent_bit not in used_bits:
                    next_line.append(parent_bit)
                    used_bits.add(parent_bit)
                    graph.add_node(parent_bit.key, subset=counter, color=color)
                    labels[parent_bit.key] = parent_bit.label
                else:
                    if not parent_bit.resolved:
                        used_prev += 1

                graph.add_edge(bit.key, parent_bit.key, color=color)

        print(f'line {counter}: {len(curr_line)}, used on prev lines: {used_prev}')
        curr_line, next_line = next_line, []
    print('Total graph', len(graph.nodes), len(graph.edges))

    print('Starting drawning')
    node_color = {
        n: data['color']
        for n, data in graph.nodes(data=True)
    }
    edge_color = {
        (n1, n2): edgedata["color"]
        for n1, n2, edgedata in graph.edges(data=True)
    }
    pos = networkx.multipartite_layout(graph, align='horizontal')
    return _graph_to_plotly(graph, pos, node_color, edge_color)
    # plt.figure(figsize=sizes)
    # networkx.draw_networkx(
    #     graph,
    #     pos=pos,
    #     with_labels=False,
    #     node_size=10,
    #     node_color=node_color,
    #     edge_color=edge_color,
    #     width=0.2,
    #     arrowsize=2,
    #     node_shape='.',
    #     labels=labels,
    #     font_size=2,
    # )
    #
    # text = networkx.draw_networkx_labels(graph, labels=labels, pos=pos, font_size=2)
    # for t in text.values():
    #     t.set_rotation(45)
    # plt.show()
    # print('Saving to file')
    # plt.savefig("bits_tree.png", dpi=500, bbox_inches='tight')
    # print('End draw_graph_parents')


def draw_graph_usages(
        start_bits: list[TreeBitAtom],
):
    print('Start draw_graph_usages')
    graph = networkx.DiGraph()
    labels = {}
    node_color_map: dict[type[TreeBitAtom], str] = {
        TreeBit: "red",
        TreeBitNOT: 'blue',
        TreeBitOR: "orange",
        TreeBitAND: 'yellow', #"olive",
        TreeBitXOR: "green",
    }

    next_line = []
    curr_line = []
    counter = 1

    for bit in start_bits:
        curr_line.append(bit)
        graph.add_node(bit.key, subset=counter,color=node_color_map[type(bit)])
        labels[bit.key] = bit.label

    used_bits = set(curr_line)
    while curr_line and counter:
        counter += 1

        used_prev = 0
        for bit in curr_line:
            bit_usages = registry[bit.key].usages
            for bit_user_key in bit_usages:
                bit_user = registry[bit_user_key].bit
                if bit_user not in used_bits:
                    next_line.append(bit_user)
                    used_bits.add(bit_user)
                    graph.add_node(bit_user.key, subset=counter, color=node_color_map[type(bit_user)])
                    labels[bit_user.key] = bit_user.label
                else:
                    if not bit_user.resolved:
                        used_prev += 1

                graph.add_edge(bit.key, bit_user.key)

        print(f'line {counter}: {len(curr_line)}, used on prev lines: {used_prev}')
        curr_line, next_line = next_line, []

    print('Starting drawning')
    pos = networkx.multipartite_layout(graph, align='horizontal')
    node_color = {
        n: data['color']
        for n, data in graph.nodes(data=True)
    }
    # edge_color = [
    #     edgedata["color"]
    #     for _, _, edgedata in graph.edges(data=True)
    # ]
    return _graph_to_plotly(graph, pos, node_color, {})
    # plt.figure(figsize=sizes)
    # networkx.draw_networkx(
    #     graph,
    #     pos=pos,
    #     with_labels=False,
    #     node_size=10,
    #     node_color=node_color,
    #     edge_color=edge_color,
    #     width=0.2,
    #     arrowsize=2,
    #     node_shape='.',
    #     labels=labels,
    #     font_size=2,
    # )
    #
    # text = networkx.draw_networkx_labels(graph, labels=labels, pos=pos, font_size=2)
    # for t in text.values():
    #     t.set_rotation(45)
    # plt.show()
    # print('Saving to file')
    # plt.savefig("bits_tree_usages.png", dpi=500, bbox_inches='tight')
    # print('End draw_graph_usages')


def draw_bit_tree(bit: TreeBitAtom):
    draw_graph_parents(predicted_h_bits=[bit])


def _graph_to_plotly(
        graph: networkx.DiGraph,
        pos,
        node_color_map,
        edge_color_map,
):
    edge_x = []
    edge_y = []

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = plotly.graph_objects.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width = 0.5),
        # marker=dict(line=dict(color=edge_color)),
        hoverinfo = 'none',
        mode = 'lines')

    node_x = []
    node_y = []
    node_colors = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_colors.append(node_color_map[node])

    node_trace = plotly.graph_objects.Scatter(
        x = node_x, y = node_y,
        mode = 'markers',
        hoverinfo = 'text',
        # color=node_color,
        marker = dict(
            color = node_colors,
            size = 6
        ),
        line_width = 1)

    fig = plotly.graph_objects.Figure(
        data=[
            edge_trace,
            node_trace,
        ],
        layout=plotly.graph_objects.Layout(
            title='<br>Тестовый граф для NTA',
            # titlefont_size = 16,
            showlegend = False,
            xaxis=dict(showgrid = False, zeroline = False, showticklabels = False),
            yaxis=dict(showgrid = False, zeroline = False, showticklabels = False)
        )
    )
    fig.show()


def scan_bits(result_bits: list[TreeBitAtom]):
    raw_bits: BitList = []
    for bit in result_bits:
        if not bit.resolved:
            raise Exception('Result bit nor resolved')
        raw_bits.append(bit.value)

    scanned_len = list_to_int(raw_bits[-64:])
    raw_bits = raw_bits[:-64]

    result_bytes = b''
    for i in range(0, scanned_len * 8, 8):
        raw_byte = list_to_int(raw_bits[i:i+8])
        result_bytes += raw_byte.to_bytes(1, 'big')

    return result_bytes


def print_tree(bits: list[TreeBitAtom]):
    next_line = []
    curr_line = bits

    used_bits = set(curr_line)
    counter = 1
    while curr_line:
        used_prev = 0
        status_line = f'{counter}. '
        for bit in curr_line:
            status_line += str(int(bit.value)) if bit.resolved else '?'

            if isinstance(bit, TreeBit):
                continue
            if isinstance(bit, TreeBitNOT):
                if bit.bit not in used_bits:
                    next_line.append(bit.bit)
                    used_bits.add(bit.bit)
                else:
                    if not bit.bit.resolved:
                        used_prev += 1
                continue
            if isinstance(bit, TreeBitOperator):
                if bit.a not in used_bits:
                    next_line.append(bit.a)
                    used_bits.add(bit.a)
                else:
                    if not bit.a.resolved:
                        used_prev += 1
                if bit.b not in used_bits:
                    next_line.append(bit.b)
                    used_bits.add(bit.b)
                else:
                    if not bit.b.resolved:
                        used_prev += 1
                continue
            raise Exception(f'Undefined TreeBit {type(bit)}')

        print(status_line)
        curr_line, next_line = next_line, []
        counter += 1


def test_sha():
    word = '1'
    # used_symbols = ['1', '2']
    used_symbols = list(string.digits)
    # used_symbols = list(string.hexdigits)
    # used_symbols = list(string.printable)
    print(f'Word: "{word}", len: {len(word)}')
    word_sha1 = sha1(word)
    real_sha1 = simple_sha1(word)
    print(f'Word SHA1: {word_sha1}')
    print(f'Real SHA1: {real_sha1}')
    assert word_sha1 == real_sha1
    print(f'Real Bits Scan (RBS): {real_bits_scan}')
    print(f'RBS X counter: {real_bits_scan.count("X")}')

    word_reverse = sha1_rev(word_sha1, len(word), used_symbols=used_symbols)
    print(f'Reverse: {word_reverse}')


def test_push():
    init_sha1_state()

    from tree_bit.base import ZERO_BIT

    input_bits = [
        TreeBit(0.5, 'ib_1'),
        TreeBit(0.5, 'ib_2'),
        TreeBit(0.5, 'ib_3'),
        TreeBit(0.5, 'ib_4'),
        TreeBit(0.5, 'ib_5'),
    ]

    exit_bit = input_bits[0]
    for input_bit in input_bits[1:]:
        exit_bit = exit_bit ^ input_bit
    exit_bits = [
        ~exit_bit
    ]

    exit_values = [ZERO_BIT]
    draw_graph(predicted_h_bits=exit_bits, sizes=(10, 10))
    ppr.PredictsPusherV2().push_presumptive_predict(exit_bits, exit_values, input_bits)


def test_operators():
    init_sha1_state()
    from tree_bit.base import ZERO_BIT, ONE_BIT, UINT_ZERO

    assert (ZERO_BIT | ZERO_BIT) == ZERO_BIT
    assert (ZERO_BIT | ONE_BIT) == ONE_BIT
    assert (ONE_BIT | ZERO_BIT) == ONE_BIT
    assert (ONE_BIT | ONE_BIT) == ONE_BIT

    assert (ZERO_BIT & ZERO_BIT) == ZERO_BIT
    assert (ZERO_BIT & ONE_BIT) == ZERO_BIT
    assert (ONE_BIT & ZERO_BIT) == ZERO_BIT
    assert (ONE_BIT & ONE_BIT) == ONE_BIT

    assert (ZERO_BIT ^ ZERO_BIT) == ZERO_BIT
    assert (ZERO_BIT ^ ONE_BIT) == ONE_BIT
    assert (ONE_BIT ^ ZERO_BIT) == ONE_BIT
    assert (ONE_BIT ^ ONE_BIT) == ZERO_BIT

    assert (~ZERO_BIT) == ONE_BIT
    assert (~ONE_BIT) == ZERO_BIT

    assert UINT_ZERO.to_int() == 0
    assert UInt32Tree.from_int(123, 'ott').to_int() == 123
    assert (UInt32Tree.from_int(1, 'one') + UInt32Tree.from_int(2, 'two')).to_int() == 3
    assert UInt32Tree.from_int(1, 'one2').rol(1).to_int() == 2
    assert UInt32Tree.from_int(1, 'one3').rol(2).to_int() == 4


if __name__ == '__main__':
    # test_operators()
    test_sha()
    # test_push()
