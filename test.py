import qutip as qt
import numpy as np
from typing import List, Tuple, Dict, Any

def solve_shortest_path(edge_list: List[Tuple[str, str, float]], source: str, destination: str) -> Dict[str, Any]:
    """
    通过转化为 QUBO 和 Ising 模型，并利用 QuTiP 求解器来寻找有向图的最短路径。
    """
    # ==========================================
    # Step 1: Topology Mapping (拓扑映射)
    # ==========================================
    nodes = set()
    for u, v, w in edge_list:
        nodes.add(u)
        nodes.add(v)
        
    N = len(edge_list)
    idx_to_edge = {i: edge for i, edge in enumerate(edge_list)}
    
    # ==========================================
    # Step 2 & 3: QUBO Matrix Construction (构建 QUBO 矩阵)
    # ==========================================
    Q = np.zeros((N, N))
    
    # 定义足够大的惩罚系数 A
    # 为了绝对压制错误路径，A 需要大于图中任何可能路径的最大总长度。
    max_weight = max(w for _, _, w in edge_list)
    A = 2 * (max_weight * N) # 更安全的下界
    qubo_offset = 0.0
    
    # 注入 H_cost (距离成本)
    for i, (_, _, w) in enumerate(edge_list):
        Q[i, i] += w
        
    # 注入 H_penalty (流量守恒惩罚)
    for k in nodes:
        # 边界条件 Delta
        delta_k = 1 if k == source else (-1 if k == destination else 0)
        
        # 找出以节点 k 为起点和终点的边索引
        E_out = [i for i, (u, _, _) in enumerate(edge_list) if u == k]
        E_in = [i for i, (_, v, _) in enumerate(edge_list) if v == k]
        
        qubo_offset += A * (delta_k ** 2)
        
        # 对角线项 (线性项 x_e^2 = x_e)
        for e in E_out:
            Q[e, e] += A - 2 * A * delta_k
        for f in E_in:
            Q[f, f] += A + 2 * A * delta_k
            
        # 交叉项: 出边与出边 (同向耦合)
        for i in range(len(E_out)):
            for j in range(i + 1, len(E_out)):
                e1, e2 = E_out[i], E_out[j]
                Q[min(e1, e2), max(e1, e2)] += 2 * A
                
        # 交叉项: 入边与入边 (同向耦合)
        for i in range(len(E_in)):
            for j in range(i + 1, len(E_in)):
                f1, f2 = E_in[i], E_in[j]
                Q[min(f1, f2), max(f1, f2)] += 2 * A
                
        # 交叉项: 出边与入边 (异向耦合)
        for e in E_out:
            for f in E_in:
                Q[min(e, f), max(e, f)] -= 2 * A

    # ==========================================
    # Step 4: Ising Parameter Extraction (提取 Ising 参数)
    # ==========================================
    h = np.zeros(N)
    J = np.zeros((N, N))
    E_offset = qubo_offset
    
    for i in range(N):
        h[i] += -Q[i, i] / 2
        E_offset += Q[i, i] / 2
        for j in range(i + 1, N):
            J[i, j] = Q[i, j] / 4
            h[i] -= Q[i, j] / 4
            h[j] -= Q[i, j] / 4
            E_offset += Q[i, j] / 4

    # ==========================================
    # Step 5: QuTiP Operator Generation (构建哈密顿量算符)
    # ==========================================
    def get_full_space_Z(e: int, num_qubits: int) -> qt.Qobj:
        """生成扩展到 2^N 维全空间的单比特 Sigma-Z 算符"""
        op_list = [qt.identity(2)] * num_qubits
        op_list[e] = qt.sigmaz()
        return qt.tensor(op_list)

    # 1. 局域场项
    H_local_ops = [h[i] * get_full_space_Z(i, N) for i in range(N) if h[i] != 0]
    # 2. 相互作用项
    H_int_ops = [J[i, j] * get_full_space_Z(i, N) * get_full_space_Z(j, N) 
                 for i in range(N) for j in range(i + 1, N) if J[i, j] != 0]
    
    # 组合算符
    I_total = qt.tensor([qt.identity(2)] * N)
    H_local = sum(H_local_ops) if H_local_ops else 0 * I_total
    H_int = sum(H_int_ops) if H_int_ops else 0 * I_total
    
    H_Ising = H_local + H_int + E_offset * I_total

    # ==========================================
    # Step 6: Solver & Post-processing Decoder (求解与解码)
    # ==========================================
    # 利用精确对角化提取本征态。返回按能量从低到高排序的特征值与特征向量
    eigenvals, eigenvecs = H_Ising.eigenstates()
    gs_energy = eigenvals[0]
    gs_vec = eigenvecs[0]
    
    # 计算基态在各个计算基（Computational Basis）上的概率
    probs = np.abs(gs_vec.full().flatten()) ** 2
    max_idx = np.argmax(probs)
    
    # 映射回二进制字符串 (QuTiP 默认 |0> 对应 +1 (即 x=0)，|1> 对应 -1 (即 x=1))
    binary_str = format(max_idx, f'0{N}b')
    
    selected_edges = []
    total_dist = 0.0
    for i, bit in enumerate(binary_str):
        if bit == '1':
            selected_edges.append(idx_to_edge[i])
            total_dist += idx_to_edge[i][2]
            
    # 拓扑路径重建与校验
    path = []
    is_valid = False
    
    if selected_edges:
        current_node = source
        path.append(current_node)
        edges_copy = selected_edges.copy()
        
        while edges_copy:
            found_next = False
            for edge in edges_copy:
                if edge[0] == current_node:
                    current_node = edge[1]
                    path.append(current_node)
                    edges_copy.remove(edge)
                    found_next = True
                    break
            
            # 路径断裂
            if not found_next:
                break
                
        # 校验：是否抵达终点，且没有游离的“幽灵环路”（即所选边被全部耗尽）
        if current_node == destination and len(edges_copy) == 0:
            is_valid = True
            
    return {
        "path": path,
        "total_distance": total_dist,
        "is_valid": is_valid,
        "ground_state_energy": gs_energy,
        "binary_state": binary_str
    }

def test_pipeline():
    print("--- 运行测试用例 1: 简单的 3 边图 ---")
    edges_simple = [
        ('S', 'A', 2.0),
        ('A', 'D', 3.0),
        ('S', 'D', 6.0)
    ]
    res1 = solve_shortest_path(edges_simple, 'S', 'D')
    print("二进制态:", res1["binary_state"])
    print("路径输出:", res1["path"])
    print("总计距离:", res1["total_distance"])
    assert res1["is_valid"], "测试 1 失败：路径违反了流量守恒！"
    assert res1["path"] == ['S', 'A', 'D'], "测试 1 失败：路径不符合预期！"
    assert abs(res1["total_distance"] - 5.0) < 1e-5, "测试 1 失败：距离计算错误！"
    print("-> 测试用例 1 通过！\n")


def test_pipeline_complex():
    print("--- 运行测试用例 2: 复杂的 6 边图 ---")
    # 构建一个包含分叉和多路径的 6 节点图
    # S -> A -> D (总权重 8)
    # S -> B -> C -> D (总权重 4) - 这是最短路径
    # S -> A -> C -> D (总权重 7)
    edges_complex = [
        ('S', 'A', 2.0),  # bit 0
        ('S', 'B', 1.0),  # bit 1
        ('A', 'C', 3.0),  # bit 2
        ('A', 'D', 6.0),  # bit 3
        ('B', 'C', 1.0),  # bit 4
        ('C', 'D', 2.0)   # bit 5
    ]
    res2 = solve_shortest_path(edges_complex, 'S', 'D')
    print("基态能量:", res2["ground_state_energy"])
    print("二进制态:", res2["binary_state"])
    print("路径输出:", res2["path"])
    print("总计距离:", res2["total_distance"])
    
    assert res2["is_valid"], "测试 2 失败：路径违反了流量守恒！"
    assert res2["path"] == ['S', 'B', 'C', 'D'], f"测试 2 失败：期望 ['S', 'B', 'C', 'D'], 实际得到 {res2['path']}"
    assert abs(res2["total_distance"] - 4.0) < 1e-5, f"测试 2 失败：期望距离 4.0，实际得到 {res2['total_distance']}"
    print("-> 测试用例 2 通过！")

def validate_against_networkx(num_trials: int = 10, max_nodes: int = 6, seed: int = 42):
    """
    用 networkx 的 Dijkstra 算法作为 ground truth，与 Ising 求解器在多个随机图上做对比。
    只对比最短距离（多条最短路径可能并存，路径本身不一定唯一）。
    """
    import networkx as nx
    import random

    rng = random.Random(seed)
    print(f"\n===== Networkx Ground-Truth 对比验证 ({num_trials} 个随机图) =====")
    num_pass = 0
    num_run = 0
    for t in range(num_trials):
        # 随机生成有向图：n 个节点，边按概率生成
        n = rng.randint(4, max_nodes)
        # S 固定在最前、D 固定在最后，中间节点随机，保证 DAG 中 S->D 总可能连通
        middle = [chr(ord('A') + i) for i in range(n - 2)]
        rng.shuffle(middle)
        node_names = ['S'] + middle + ['D']
        source, dest = 'S', 'D'

        G = nx.DiGraph()
        G.add_nodes_from(node_names)
        edge_list = []
        # 生成候选边，保证是 DAG（按节点顺序连边，避免环路让最短路径问题更清晰）
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    u, v = node_names[i], node_names[j]
                    w = round(rng.uniform(1.0, 9.0), 1)
                    G.add_edge(u, v, weight=w)
                    edge_list.append((u, v, w))

        # 强制保证至少存在一条 S->D 路径，否则跳过该图
        if not nx.has_path(G, source, dest):
            continue
        num_run += 1

        # networkx ground truth (Dijkstra)
        nx_dist = nx.dijkstra_path_length(G, source, dest)
        nx_path = nx.dijkstra_path(G, source, dest)

        # Ising 求解器
        res = solve_shortest_path(edge_list, source, dest)
        ising_dist = res["total_distance"] if res["is_valid"] else float('inf')

        match = abs(ising_dist - nx_dist) < 1e-5 and res["is_valid"]
        status = "PASS" if match else "FAIL"
        print(f"[Trial {t+1}] {status} | nodes={n}, edges={len(edge_list)} | "
              f"nx_dist={nx_dist} ({nx_path}) | ising_dist={ising_dist} ({res['path']})")
        if match:
            num_pass += 1
        else:
            print(f"    -> binary_state={res['binary_state']}, energy={res['ground_state_energy']}")
            print(f"    -> edges={edge_list}")

    print(f"\n总计: {num_pass}/{num_run} 通过 (共生成 {num_trials} 个图，{num_trials - num_run} 个不连通被跳过)")
    assert num_run > 0, "没有可用的连通图用于验证！"
    assert num_pass == num_run, "存在与 networkx 不一致的测试用例！"
    print("-> 全部随机图与 networkx ground truth 一致！")


def visualize_comparison(edge_list: List[Tuple[str, str, float]],
                         source: str,
                         dest: str,
                         output_file: str = "shortest_path_comparison.png"):
    """
    可视化对比图：
    - 黑色绘制原图（节点、边、权重）
    - 绿色标注 networkx Dijkstra 的 ground truth 路径
    - 蓝色标注 Ising 求解器输出路径
    为避免两路径重合时互相遮挡，绿/蓝边使用相反方向的曲率。
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib as mpl

    # 设置支持中文的字体（Windows 优先使用 Microsoft YaHei / SimHei）
    for font in ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']:
        try:
            mpl.font_manager.findfont(font, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            break
        except Exception:
            continue

    # 构图
    G = nx.DiGraph()
    for u, v, w in edge_list:
        G.add_edge(u, v, weight=w)

    # Ground truth (Dijkstra)
    nx_path = nx.dijkstra_path(G, source, dest)
    nx_dist = nx.dijkstra_path_length(G, source, dest)
    nx_edges = list(zip(nx_path[:-1], nx_path[1:]))

    # Ising 求解器输出
    res = solve_shortest_path(edge_list, source, dest)
    ising_path = res["path"] if res["is_valid"] else []
    ising_dist = res["total_distance"] if res["is_valid"] else float('inf')
    ising_edges = list(zip(ising_path[:-1], ising_path[1:])) if len(ising_path) > 1 else []

    # 分层布局：按 BFS 距离 source 的层数左→右排列
    distances = nx.single_source_shortest_path_length(G, source)
    layers = {}
    for node, d in distances.items():
        layers.setdefault(d, []).append(node)
    max_layer = max(layers.keys()) if layers else 1
    pos = {}
    for layer, nodes_in_layer in layers.items():
        for i, node in enumerate(sorted(nodes_in_layer)):
            x = layer / max_layer if max_layer > 0 else 0.5
            y = (i - (len(nodes_in_layer) - 1) / 2) / max(len(nodes_in_layer), 1)
            pos[node] = (x, y)

    fig, ax = plt.subplots(figsize=(11, 7))

    # 黑色绘制原图：节点（黑底白字）、边
    nx.draw_networkx_nodes(G, pos, node_color='black', node_size=800, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', font_size=12, font_weight='bold', ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='black', width=1.5,
                           arrows=True, arrowstyle='-|>', arrowsize=18,
                           connectionstyle="arc3,rad=0", ax=ax)

    # 权重标签：手动放置，沿边方向交替位置 + 垂直偏移，避免重叠
    for idx, (u, v, w) in enumerate(edge_list):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        # 沿边方向交替取 0.35 / 0.65 位置
        t = 0.35 if idx % 2 == 0 else 0.65
        mx, my = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        # 垂直方向偏移
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            px, py = -dy / length, dx / length
        else:
            px, py = 0, 0
        offset = 0.05 * ((idx % 3) - 1)  # -0.05 / 0 / 0.05
        lx, ly = mx + px * offset, my + py * offset
        ax.text(lx, ly, f"{w}", fontsize=11, color='black', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.85),
                zorder=5)

    # 绿色叠加 ground truth：点线，无箭头
    if nx_edges:
        nx.draw_networkx_edges(G, pos, edgelist=nx_edges, edge_color='green', width=3.5,
                               arrows=False, ax=ax, alpha=0.9)

    # 蓝色叠加算法输出：虚线，无箭头
    if ising_edges:
        nx.draw_networkx_edges(G, pos, edgelist=ising_edges, edge_color='blue', width=3.5,
                               arrows=False, style='--', ax=ax, alpha=0.9)

    # 图例（精简，不含结果，使用线条样式展示点线/虚线）
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='原图'),
        Line2D([0], [0], color='green', lw=3.5, linestyle=':', label='Ground Truth'),
        Line2D([0], [0], color='blue', lw=3.5, linestyle='--', label='Ising 求解器'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.95)

    ax.set_title(f"最短路径对比可视化: {source} → {dest}", fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存至: {output_file}")
    print(f"  Ground Truth (Dijkstra): {nx_path}, 距离={nx_dist}")
    print(f"  Ising 求解器:           {ising_path}, 距离={ising_dist}")
    return output_file


if __name__ == "__main__":
    test_pipeline()
    test_pipeline_complex()
    validate_against_networkx(num_trials=10, max_nodes=6, seed=42)

    # 可视化对比（使用复杂测试图）
    print("\n===== 生成可视化对比图 =====")
    edges_complex = [
        ('S', 'A', 2.0),
        ('S', 'B', 1.0),
        ('A', 'C', 3.0),
        ('A', 'D', 6.0),
        ('B', 'C', 1.0),
        ('C', 'D', 2.0)
    ]
    visualize_comparison(edges_complex, 'S', 'D', output_file="shortest_path_comparison.png")