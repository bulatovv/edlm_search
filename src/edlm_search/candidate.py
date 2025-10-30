from edlm_search.llm_pipeline import LLMPipeline
from edlm_search.problem import Problem


class Candidate:
    def __init__(self, files: dict[str, str], idea: str):
        self.files: dict[str, str] = files
        self.idea = idea

    @classmethod
    async def new_from_problem(cls, problem: Problem, llm_pipeline: LLMPipeline):
        idea, files = await llm_pipeline.generate_files_from_template(
            'new_candidate', problem=problem
        )
        return Candidate(files, idea)
