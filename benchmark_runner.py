from benchmark import (
    Benchmark,
    Ranger,
)
from datasets import load_dataset
import os


def upload_legalbench() -> None:
    # load dataset from hugginface or local
    dataset = load_dataset(
        "legalbench/definition_classification/",
        "ssla_company_defendents/",  # TODO
        delimiter="\t",
        column_names=["text", "label"],
    )

    legalbench = Benchmark(name="legalbench")

    def my_parser(input, **kwargs):
        baseprompt = kwargs["template"]
        input = kwargs["input"]
        return baseprompt.replace("{{input}}", input)

    legalbench.generate_from_template = my_parser
    legalbench.add_dataset("abc", dataset)

    f = open(
        "legalbench/supply_chain_disclosure_best_practice_certification/base_prompt.txt",
        "r",
    )

    template = f.read()

    legalbench.add_assignment(
        "supply_chain_disclosure_best_practice",
        dataset_name="abc",
        input_col="text",
        output_col="answer",
        template=template,
    )

    Ranger.upload_benchmark(legalbench, "ranger-uploads", "legalbench")


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
    print(ranger.get_results()[0])


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
    print(ranger.get_results()[0])


# TEST
def cloud_model_local_data():
    model_source = "gpt2"
    data_csv_location = "dummy_data.csv"

    # create ranger
    #TODO: pass key in config file, list avalaible models from base10 perhaps
    ranger: Ranger = Ranger(model_source, "baseten", "aaa", "bbb")

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
    print(ranger.get_results()[0])


# TEST
def cloud_model_hf_data():
    return

#TODO: save results from a run on AWS and associate with benchmark_run Id. 
def save_results_aws():
    return

# TEST
def legalbench() -> None:
    model_source = "NousResearch/Nous-Hermes-Llama2-13b"

    ranger: Ranger = Ranger(model_source)

    custom_benchmark: Benchmark = Benchmark("legalbench")

    # add dataset, assignment and baseprompt for each test tsv and txt in legalbench folder
    for assignment_name in os.listdir("legalbench"):
        if os.path.isdir(f"legalbench/{assignment_name}"):
            for file in os.listdir(f"legalbench/{assignment_name}"):
                if file.endswith("test.tsv"):
                    dataset_name = file
                    custom_benchmark.add_dataset_from_csv(
                        dataset_name,
                        f"legalbench/{assignment_name}/{file}",
                        delimiter="\t",
                    )
                    custom_benchmark.add_assignment(
                        f"{dataset_name}_assignment",
                        dataset_name,
                        "text",
                        "answer",
                    )
                elif file.endswith(".txt"):
                    with open(f"legalbench/{assignment_name}/{file}", "r") as f:
                        template = f.read()
                    for assignment in custom_benchmark.assignments:
                        if assignment.name == f"{dataset_name}_assignment":
                            assignment.template = template

    ranger.add_benchmark(custom_benchmark)
    ranger.run_benchmark("legalbench")

    print(ranger.get_results()[0])


if __name__ == "__main__":
    # local_model_local_data()
    # local_model_hf_dataset_()
    #cloud_model_local_data()
    # cloud_model_hf_data()
    # upload_legalbench()
    legalbench()
