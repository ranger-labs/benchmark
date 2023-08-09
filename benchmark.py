from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import TextGenerationPipeline, pipeline
from transformers.pipelines.base import KeyDataset
from datasets import Dataset, load_dataset
from typing import Callable
from tqdm import tqdm
from enum import Enum
import logging as log
import json
import boto3


# TODO: This should be using logging instead of print but
class Colors(Enum):
    GREEN = 1
    BLUE = 2


def colored_print(*args, color: Colors = Colors.GREEN):
    match color:
        case Colors.GREEN:
            print("\x1b[6;30;42m", *args, "\x1b[0m")


class RangerException(Exception):
    def __init__(self, error_type: str, message: str):
        super().__init__(error_type + ": " + message)


class BenchmarkResult:
    def __init__(self, benchmark_name: str, number_of_assignments: int):
        self.benchmark_name: str = benchmark_name
        self.number_of_assignments: int = number_of_assignments

        self.results: dict[str, float] = {}
        self.average: float = -1

    def add_result(self, assignment_name: str, result: float) -> None:
        self.results[assignment_name] = result

    def compute_average(self) -> float:
        total = 0
        for result in self.results.values():
            total += result

        self.average = total / self.number_of_assignments
        return self.average

    def __str__(
        self,
    ):
        return f"""
        Benchmark: {self.benchmark_name}
        Number of Assignments: {self.number_of_assignments}
        Average: {self.average}
        Results: {self.results}
        """


class Model:
    def __init__(self, name: str, model_source: str, max_tokens: int, key, _id):
        self.model_source: str = model_source
        self.max_tokens: int = max_tokens

        match model_source:
            case "huggingface":
                log.debug("Using pipeline")
                self.pipeline: TextGenerationPipeline = pipeline(
                    "text-generation", name, trust_remote_code=True  # type: ignore[code]
                )
            case "baseten":
                log.debug("Using baseten")
                if not key:
                    raise RangerException("MISSING", "No baseten credentials")
                if not _id:
                    raise RangerException("MISSING", "No baseten model identification")

                self.key: str = key
                self._id: str = _id

            case _:
                raise RangerException("INVALID", "Invalid model source")

    def run(self, assignment: Assignment) -> None:
        match self.model_source:
            case "huggingface":
                self.local_model(assignment)
            case "baseten":
                self.baseten(assignment)

    def local_model(self, assignment: Assignment) -> None:
        prompts: Dataset = assignment.get_input_dataset()
        answers: list[str] = assignment.get_outputs()

        if not len(prompts) == len(answers):
            raise RangerException("MISMATCH", "Prompt and Anwer length do not match")

        for output in tqdm(
            self.pipeline(
                KeyDataset(prompts, assignment.input_col),  # type: ignore[code]
                pad_token_id=50256,
                max_new_tokens=self.max_tokens,
                do_sample=True,
                truncation=True,
                max_length=512,
            )
        ):
            # colored_print(output)
            # TODO it might not be "generated_text" for all models?
            assignment.outputs.append(output[0]["generated_text"])  # type: ignore[code]

        if not len(prompts) == len(assignment.outputs):
            raise RangerException("MISMATCH", "Prompt and Output length do not match")

        for i, output in enumerate(assignment.outputs):
            comparison = assignment.comparison_function(output, answers[i])
            assignment.result += int(comparison)

        assignment.result = assignment.result / len(assignment.outputs)

    def baseten(self, assignment: Assignment) -> None:
        prompts: list[str] = assignment.get_inputs()
        answers: list[str] = assignment.get_outputs()

        if not len(prompts) == len(answers):
            raise RangerException("MISMATCH", "Prompt and Anwer length do not match")

        import models

        def run_and_compare(prompt: str, answer: str, comparison_function) -> bool:
            try:
                # TODO models currently takes in name of model,
                # we want it to take in API_KEY and MODEL_ID
                log.info(prompt)
                output = models.run_model(prompt, "falcon", self.max_tokens)
                assignment.outputs.append(output)
            except models.APIError as e:
                print(e)
                return False

            return comparison_function(output, answer)

        NUMBER_OF_THREADS = 20
        with tqdm(
            total=len(prompts),
            colour="green",
            desc="Processing",
            # leave=False,
        ) as progress_bar:
            # thread pool executor
            with ThreadPoolExecutor(max_workers=NUMBER_OF_THREADS) as executor:
                futures = {
                    executor.submit(
                        run_and_compare,
                        prompt=prompt,
                        answer=answers[i],
                        comparison_function=assignment.comparison_function,
                    ): (i, prompt)
                    for (i, prompt) in enumerate(prompts)
                }

                # update loading bar and increase valid reuslt count
                for future in as_completed(futures):
                    assignment.result += future.result()
                    progress_bar.update(1)
        assignment.result = assignment.result / len(prompts)


class Assignment:
    @staticmethod
    def sample_comparison_function(output: str, answer: str) -> bool:
        output = str(output).lower()
        answer = str(answer).lower()

<<<<<<< HEAD
        # output = re.sub("\n", "", output)
        # answer = re.sub("\n", "", answer)

        output = re.sub("[^A-Za-z0-9]+", "", output.lower())
        answer = re.sub("[^A-Za-z0-9]+", "", answer.lower())

=======
>>>>>>> d9ef30c8029220bb2bfe4b2c9dbd5f896788b366
        log.debug(f"Comparing output: '{output}' to answer: '{answer}'")

        return output == answer or output.startswith(answer)

    def __init__(
        self,
        name: str,
        dataset_name: str,
        input_col: str,
        output_col: str,
        comparison_function: Callable[[str, str], bool],
        template=None,
        generate_from_template=None,
    ):
        self.name: str = name
        self.dataset_name: str = dataset_name
        self.input_col: str = input_col
        self.output_col: str = output_col
        self.comparison_function: Callable[[str, str], bool] = comparison_function

        self.result: float = 0
        self.outputs: list[str] = []

        # dataset is not included at initialization,
        # it is added when the assignment is added to the benchmark
        self.dataset: Dataset

        # TODO not implementing for now
        if template and generate_from_template:
            self.template = template
            self.generate_from_template = generate_from_template

    def add_dataset(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def get_result(self) -> float:
        return self.result

    def get_input_dataset(self) -> Dataset:
        return self.dataset.select_columns(self.input_col)

    def get_output_dataset(self) -> Dataset:
        return self.dataset.select_columns(self.output_col)

    def get_inputs(self) -> list[str]:
        return self.dataset[self.input_col]

    def get_outputs(self) -> list[str]:
        return self.dataset[self.output_col]

    def run(self, model: Model) -> float:
        model.run(self)
        return self.result


class Benchmark:
    # creates Benchmark from a json
    @staticmethod
    def from_json(json_str: str) -> Benchmark:
        # loads json
        data = json.loads(json_str)

        # assigns values
        benchmark = Benchmark(data["name"])
        benchmark.assignments = data["assignments"]

        # Load datasets into the benchmark
        for dataset_name, dataset in data["datasets"].items():
            benchmark.add_dataset(
                benchmark.datasets[dataset_name], Dataset.from_dict(dataset)  # type: ignore[code]
            )
        return benchmark

    def __init__(self, name: str, preset: bool = False):
        self.name: str = name
        self.assignments: list[Assignment] = []
        self.datasets: dict[str, Dataset] = {}

        self.result: BenchmarkResult

        if preset:
            self.assigments = [
                "ssla_company_defendents"  # TODO
            ]  # add all tasks here, not just this example one ex (load from s3 bucket json?)

        colored_print("Created benchmark: ", self)

    # adds a dataset
    def add_dataset(self, dataset_name: str, dataset: Dataset):
        # TODO: Use our own dataset table
        # which maps a dataset name to hugging face, file on s3, etc.
        # When the dataset is added, add it to the database
        self.datasets[dataset_name] = dataset

        colored_print("Created and added dataset: ", self.datasets[dataset_name])

    # Dataset.from can take csv or json as argument if we want

    # adds a csv as a dataset
    def add_dataset_from_csv(self, dataset_name: str, data_csv_location: str, **kwargs):
        # assuming from_csv returns Dataset
        dataset: Dataset = Dataset.from_csv(data_csv_location, **kwargs)  # type: ignore[code]
        self.add_dataset(dataset_name, dataset)

    # adds a json as a dataset
    def add_dataset_from_json(self, dataset_name: str, dataset_source: str, **kwargs):
        # assuming from_json returns Dataset
        dataset: Dataset = Dataset.from_json(dataset_source, **kwargs)  # type: ignore[code]
        self.add_dataset(dataset_name, dataset)

    # adds a huggingface dataset
    def add_dataset_from_hf(self, dataset_name: str, dataset_source: str, **kwargs):
        # assuming load_dataset returns Dataset
        dataset: Dataset = load_dataset(
            dataset_source, split="train", **kwargs
        )  # type: ignore[code]
        self.add_dataset(dataset_name, dataset)

    # adds an assignment
    def add_assignment(
        self,
        name: str,
        dataset_name: str,
        input_col: str,
        output_col: str,
        template=None,
        generate_from_template=None,
        comparison_function=Assignment.sample_comparison_function,
    ):
        # to_add_ds = self.datasets[dataset_name]
        self.assignments.append(
            Assignment(
                name,
                dataset_name,
                input_col,
                output_col,
                comparison_function,
                template,
                generate_from_template,
            )
        )

        try:
            self.assignments[-1].add_dataset(self.datasets[dataset_name])
        except Exception:
            raise RangerException("NOT FOUND", f"Dataset {dataset_name} not found")

        colored_print("Created and added assignment: ", self.assignments[-1])

    def get_result(self) -> BenchmarkResult:
        return self.result

    def run(self, model: Model):
        """Runs all assignments from this benchmark"""
        self.result = BenchmarkResult(self.name, len(self.assignments))

        for assignment in self.assignments:
            try:
                assignment_result = assignment.run(model)
                self.result.add_result(assignment.name, assignment_result)
<<<<<<< HEAD
            except:
                print(assignment.name, " failed")
=======
            except Exception:
                log.debug(f"Assignment {assignment.name} failed")
>>>>>>> d9ef30c8029220bb2bfe4b2c9dbd5f896788b366

        self.result.compute_average()

    # TODO ?
    def template_parser(self, **kwargs):
        baseprompt = kwargs["template"]
        input = kwargs["input"]
        return baseprompt.replace("{{abc}}", input)

    def to_json(self) -> str:
        data = {
            "name": self.name,
            "assignments": [assignment.__dict__ for assignment in self.assignments],
            "datasets": {
                name: dataset.features for name, dataset in self.datasets.items()
            },
        }

        colored_print(data)
        return json.dumps(data)


class Ranger:
    # initializes Ranger
    def __init__(
        self,
        name: str,
        model_source: str = "huggingface",
        max_tokens: int = 5,
        key=None,
        _id=None,
    ):
        self.model = Model(name, model_source, max_tokens, key, _id)

        self.benchmarks: list[Benchmark] = []
        self.aws_access_key_id: str = "secret"
        self.aws_secret_access_key: str = "secret"

        colored_print("Created ranger: ", self)

    # helper function, finds a benchmark by name
    def __find_benchmark_by_name(self, benchmark_name) -> Benchmark:
        for benchmark in self.benchmarks:
            if benchmark.name == benchmark_name:
                return benchmark

        raise RangerException(
            "NOT FOUND", f"No benchmark with the name {benchmark_name} found"
        )

    # adds a benchmark given the name or object
    def add_benchmark(self, benchmark: str | Benchmark) -> None:
        if isinstance(benchmark, Benchmark):
            self.benchmarks.append(benchmark)
            return

        # creates default benchmark with all assignemnts
        # TODO idk if load_benchmark works perfect, ask Jake
        created_benchmark = self.load_benchmark(benchmark)
        self.benchmarks.append(created_benchmark)

    def add_many_benchmarks(self, benchmarks: list[str] | list[Benchmark]) -> None:
        for benchmark in benchmarks:
            self.add_benchmark(benchmark)

    # returns the result of the specified benchmark
    def get_benchmark_result(self, benchmark_name: str) -> BenchmarkResult:
        benchmark = self.__find_benchmark_by_name(benchmark_name)
        return benchmark.get_result()

    def get_results(self) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []

        for benchmark in self.benchmarks:
            results.append(self.get_benchmark_result(benchmark.name))

        return results

    # runs specified benchmark
    def run_benchmark(self, benchmark_name: str) -> None:
        # TODO add baseten option
        benchmark = self.__find_benchmark_by_name(benchmark_name)
        benchmark.run(self.model)

    def run_all(self) -> None:
        for benchmark in self.benchmarks:
            benchmark.run(self.model)

    # loads a benchmark from an S3 bucket, ask Jake
    def load_benchmark(self, benchmark_name: str) -> Benchmark:
        s3 = boto3.client(
            "s3",
            self.aws_access_key_id,
            self.aws_secret_access_key,
        )
        # TODO: cross check database first to make sure benchmark exists :)
        response = s3.get_object(Bucket="ranger-uploads", Key=benchmark_name)
        json_str = response["Body"].read().decode("utf-8")
        return Benchmark.from_json(json_str)

    # uploads a benchmark to an S3 bucket, ask Jake
    def upload_benchmark(
        self, benchmark: Benchmark, bucket_name: str, s3_key: str
    ) -> None:
        # TODO: create database record detailing the new benchmark
        # TODO: we need to hash database / check for duplicate uploads some way. (name?)
        s3 = boto3.client("s3", self.aws_access_key_id, self.aws_secret_access_key)
        s3.put_object(Body=benchmark.to_json(), Bucket=bucket_name, Key=s3_key)
