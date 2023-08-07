from benchmark import Benchmark, Ranger
import logging as log


# TEST
def local_model_local_data() -> None:
    """BASIC TEST"""
    model_source = "gpt2"
    data_csv_location = "dummy_data.csv"

    # create ranger
    ranger: Ranger = Ranger(model_source)

    # create benchmark
    custom_benchmark: Benchmark = Benchmark("my_custom_benchmark")

    # add csv dataset to benchmark
    custom_benchmark.add_dataset_from_csv("dummy_dataset", data_csv_location)

    # add custom assignment to benchmark
    custom_benchmark.add_assignment(
        "custom_assignment", "dummy_dataset", "text", "answer"
    )

    # add benchmark to ranger
    ranger.add_benchmark(custom_benchmark)

    # run and retrieve results
    ranger.run_all()
    print(ranger.get_results())


# TEST
def local_model_hf_dataset_() -> None:
    model_source = "gpt2"

    ranger: Ranger = Ranger(model_source)
    custom_benchmark: Benchmark = Benchmark("my_custom_benchmark")

    custom_benchmark.add_dataset_from_hf(
        "bugged_dataset", "NeuroDragon/BuggedPythonLeetCode"
    )

    # define custom comparison function
    def my_comparison(a: str, b: str) -> bool:
        # TODO make an actual comparison
        return True

    custom_benchmark.add_assignment(
        "bugged_assignemnt",
        "bugged_dataset",
        "question",
        "answer",
        comparison_function=my_comparison,
    )

    ranger.add_benchmark(custom_benchmark)
    ranger.run_all()
    print(ranger.get_results())


# TEST
def cloud_model_local_data():
    model_source = "gpt2"
    data_csv_location = "dummy_data.csv"

    ranger: Ranger = Ranger(model_source, "baseten", "aaa", "bbb")
    custom_benchmark: Benchmark = Benchmark("my_custom_benchmark")

    custom_benchmark.add_dataset_from_csv("dummy_dataset", data_csv_location)

    custom_benchmark.add_assignment(
        "custom_assignment", "dummy_dataset", "text", "answer"
    )

    ranger.add_benchmark(custom_benchmark)
    ranger.run_all()

    print(ranger.get_results())


# TEST
def cloud_model_hf_data():
    return


if __name__ == "__main__":
    log.basicConfig(level=log.DEBUG, filename="benchmark_runner.log")
    local_model_local_data()
    # local_model_hf_dataset_()
    # cloud_model_local_data()
    # cloud_model_hf_data()
