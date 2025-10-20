from edlm_search.llm_pipeline import LLMPipeline
from edlm_search.problem import Problem


class Candidate:
    def __init__(self): ...

    @classmethod
    async def from_problem(cls, problem: Problem, llm_pipeline: LLMPipeline):
        files = await llm_pipeline.generate_files_from_template(
            'new_candidate', problem=problem
        )
        print(files)
