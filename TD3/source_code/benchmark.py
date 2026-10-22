from algorithms import alphabeta, mcts, minimax
from game import create_board
import matplotlib.pyplot as plt

#1
def benchmark(board, algorithm, **kwargs):
	result = algorithm(board, **kwargs)
	return {
		"algorithm": algorithm.__name__,
		"move": result["move"],
		"score": result["score"],
		"nodes_explored": result["nodes_explored"],
		"time": result["time"],}

#2
def compare_minimax_alphabeta():
	board = create_board()
	results = []

	print(f"{'Algorithm':<12} {'Depth':>5} {'Move':>5} {'Score':>10} {'Nodes':>10} {'Time (s)':>10}")
	print("-" * 64)

	for depth in range(1, 6):
		minimax_result = benchmark(board, minimax, depth=depth)
		alphabeta_result = benchmark(board, alphabeta, depth=depth)

		if minimax_result["score"] != alphabeta_result["score"]:
			raise AssertionError(f"Scores differ at depth {depth}")
		if alphabeta_result["nodes_explored"] > minimax_result["nodes_explored"]:
			raise AssertionError(f"Alpha-Beta explored more nodes at depth {depth}")

		for name, result in (("Minimax", minimax_result), ("Alpha-Beta", alphabeta_result)):
			print(
				f"{name:<12} {depth:>5} {str(result['move']):>5} "
				f"{result['score']:>10} {result['nodes_explored']:>10} "
				f"{result['time']:>10.6f}")
			results.append({"depth": depth, **result})

	return results

#3
def compare_mcts_simulations():
	board = create_board()
	results = []
	simulation_budgets = (100, 500, 1000, 2000)

	print(f"{'Algorithm':<12} {'Simulations':>11} {'Move':>5} {'Reward':>10} {'Nodes':>10} {'Time (s)':>10}")
	print("-" * 64)

	for simulations in simulation_budgets:
		result = benchmark(board, mcts, num_simulations=simulations, seed=42)
		print(
			f"{'MCTS':<12} {simulations:>11} {str(result['move']):>5} "
			f"{result['score']:>10.4f} {result['nodes_explored']:>10} "
			f"{result['time']:>10.6f}")
		results.append({"simulations": simulations, **result})

	return results

#4
def run_all_benchmarks():
	board = create_board()
	results = []
	minimax_by_depth = {}

	for algorithm, configurations in (
		(minimax, (("Depth 1", {"depth": 1}), ("Depth 2", {"depth": 2}),
		           ("Depth 3", {"depth": 3}), ("Depth 4", {"depth": 4}),
		           ("Depth 5", {"depth": 5}))),
		(alphabeta, (("Depth 1", {"depth": 1}), ("Depth 2", {"depth": 2}),
		             ("Depth 3", {"depth": 3}), ("Depth 4", {"depth": 4}),
		             ("Depth 5", {"depth": 5}))),
		(mcts, (("100 simulations", {"num_simulations": 100, "seed": 42}),
		        ("500 simulations", {"num_simulations": 500, "seed": 42}),
		        ("1000 simulations", {"num_simulations": 1000, "seed": 42}),
		        ("2000 simulations", {"num_simulations": 2000, "seed": 42}))),):
		
		for configuration, parameters in configurations:
			result = benchmark(board, algorithm, **parameters)
			row = {
				"algorithm": "Alpha-Beta" if algorithm is alphabeta else algorithm.__name__.title(),
				"configuration": configuration,
				"depth": parameters.get("depth"),
				"simulations": parameters.get("num_simulations"),
				"move": result["move"],
				"nodes_explored": result["nodes_explored"],
				"time": result["time"],}
			
			if algorithm is minimax:
				minimax_by_depth[parameters["depth"]] = result
			elif algorithm is alphabeta:
				depth = parameters["depth"]
				if result["score"] != minimax_by_depth[depth]["score"]:
					raise AssertionError(f"Scores differ at depth {depth}")
				if result["nodes_explored"] > minimax_by_depth[depth]["nodes_explored"]:
					raise AssertionError(f"Alpha-Beta explored more nodes at depth {depth}")
			results.append(row)

	print(f"{'Algorithm':<12} {'Configuration':<20} {'Move':>5} {'Nodes':>10} {'Time (s)':>10}")
	print("-" * 63)
	for row in results:
		print(
			f"{row['algorithm']:<12} {row['configuration']:<20} "
			f"{str(row['move']):>5} {row['nodes_explored']:>10} {row['time']:>10.6f}")

	plot_benchmark_results(results)
	return results

#5
def plot_benchmark_results(results):

	minimax_results = [row for row in results if row["algorithm"] == "Minimax"]
	alphabeta_results = [row for row in results if row["algorithm"] == "Alpha-Beta"]
	mcts_results = [row for row in results if row["algorithm"] == "Mcts"]

	plt.figure("Minimax and Alpha-Beta nodes")
	plt.plot(
		[row["depth"] for row in minimax_results],
		[row["nodes_explored"] for row in minimax_results],
		marker="o", label="Minimax")
	plt.plot(
		[row["depth"] for row in alphabeta_results],
		[row["nodes_explored"] for row in alphabeta_results],
		marker="o", label="Alpha-Beta")
	plt.xlabel("Search depth")
	plt.ylabel("Explored nodes")
	plt.title("Explored nodes by search depth")
	plt.xticks([row["depth"] for row in minimax_results])
	plt.grid(True, linestyle=":", alpha=0.6)
	plt.legend()
	plt.tight_layout()

	plt.figure("MCTS execution time")
	plt.plot(
		[row["simulations"] for row in mcts_results],
		[row["time"] for row in mcts_results],
		marker="o", color="darkorange", label="MCTS")
	plt.xlabel("Number of simulations")
	plt.ylabel("Execution time (s)")
	plt.title("MCTS execution time by simulation budget")
	plt.xticks([row["simulations"] for row in mcts_results])
	plt.grid(True, linestyle=":", alpha=0.6)
	plt.legend()
	plt.tight_layout()
	plt.show()


if __name__ == "__main__":
	run_all_benchmarks()
