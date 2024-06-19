import pandas as pd
import matplotlib.pyplot as plt


class ResultsHolder:
    def __init__(self):
        self.data = pd.DataFrame(columns=["id", "name", "f_name", "group", "value"])
        self.counter = {}
        self.lower_better = []

    def clear(self):
        self.data = pd.DataFrame(columns=["id", "name", "f_name", "group", "value"])
        self.counter = {}

    def save(self, path):
        self.data.to_csv(path)

    def add_lower_better(self, *f_names):
        for f_name in f_names:
            if f_name not in self.lower_better:
                self.lower_better.append(f_name)

    def new_data(self, name, f_name, group, value):
        counter = self.get_counter(name) + 1
        self.counter[name] = counter
        self.data.loc[len(self.data)] = [counter, name, f_name, group, value]

    def new_datas(self, name, datas):
        counter = self.get_counter(name) + 1
        self.counter[name] = counter
        for f_name, group, value in datas:
            self.data.loc[len(self.data)] = [counter, name, f_name, group, value]

    def get_counter(self, name):
        return self.counter.get(name, 0)

    def get_new_results(self, name):
        results = []
        match_data = self.data.query(f"name=='{name}'")
        for f_name, data in match_data.groupby("f_name"):
            new_id = data.loc[data.index[-1], "id"]
            new_value = data.loc[data.index[-1], "value"]
            group = data.loc[data.index[-1], "group"]
            results.append((new_id, f_name, group, new_value))
        return results

    def get_best(self, name):
        bests = {}
        match_data = self.data.query(f"name=='{name}'").copy()
        weight_mapping = {}
        for m in match_data["f_name"].unique():
            if m in self.lower_better:
                weight_mapping[m] = -1
            else:
                weight_mapping[m] = 1
        match_data["weighted_value"] = match_data["f_name"].map(weight_mapping) * match_data["value"]
        bests["summary"] = match_data.groupby("id")["weighted_value"].sum().idxmax()
        for f_name, data in match_data.groupby("f_name"):
            if f_name in self.lower_better:
                best_id = data.loc[data["value"].idxmin(), "id"]
            else:
                best_id = data.loc[data["value"].idxmax(), "id"]
            bests[f_name] = best_id
        return bests

    def plot(self, name, save_path):
        match_data = self.data.query(f"name=='{name}'")
        bests = self.get_best(name)
        fig, axes = plt.subplots(match_data["group"].value_counts().count())
        if isinstance(axes, plt.Axes):
            axes = [axes]
        for i, (group, g_data) in enumerate(match_data.groupby("group")):
            ax = axes[i]
            for f_name, data in g_data.groupby("f_name"):
                best_id = bests[f_name]
                ax.plot(data["id"], data["value"], label=f_name)
                ax.scatter([best_id], data[data["id"] == best_id]["value"])
            ax.legend()
        fig.savefig(save_path)
        plt.close(fig)
