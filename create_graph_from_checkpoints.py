import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Logs input dir")
    args = parser.parse_args()

    args.input_dir = Path(args.input_dir)

    logs_name = args.input_dir.name

    gen = 0
    maxs = []
    avgs = []
    for log in tqdm(sorted(args.input_dir.rglob("epoch-log*"))):
        with open(log, "rb") as f:
            data = pickle.load(f)
        for i, d in enumerate(data):
            if i == 0 and gen != 0:
                continue
            maxs.append(d["max"])
            avgs.append(d["mean"])
            gen += 1

    plt.plot(range(gen), maxs, label="Maximum")
    plt.plot(range(gen), avgs, label="Average")
    plt.title(f"{logs_name},gen={gen-1}")
    plt.legend()
    plt.xlabel("Number of generations")
    plt.ylabel("Score")

    output_file = args.input_dir / "max-average-plot.png"
    plt.savefig(output_file)
    print(f"Saved graph to '{output_file}'")
